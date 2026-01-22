"""
FAISS 索引管理
支持单个索引文件和按文档分别存储的索引文件
"""
import faiss
import numpy as np
import pickle
import logging
from typing import List, Tuple, Optional, Dict
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class FAISSIndex:
    """FAISS 索引管理类，用于高效的相似度搜索"""
    
    def __init__(self, index_path: str = None, index_dir: str = None):
        """
        初始化 FAISS 索引
        
        Args:
            index_path: 单个索引文件路径（用于全局索引模式）
            index_dir: 索引文件目录（用于按文档存储模式）
        """
        self.index_path = index_path or os.getenv(
            "FAISS_INDEX_PATH", 
            "./data/index/faiss.index"
        )
        self.index_dir = index_dir or os.getenv(
            "FAISS_INDEX_DIR",
            None
        )
        self.id_map_path = self.index_path + ".ids" if self.index_path else None
        self.index = None
        self.chunk_ids = []
        self.document_indices: Dict[str, faiss.Index] = {}  # 按文档存储的索引
        self.document_chunk_maps: Dict[str, List[str]] = {}  # 每个文档的 chunk_id 映射
        
        # 如果指定了 index_dir，则使用目录模式
        if self.index_dir:
            self._load_document_indices()
        elif os.path.exists(self.index_path):
            self.load()
    
    def _load_document_indices(self):
        """从目录加载所有文档的索引"""
        index_dir = Path(self.index_dir)
        logger.info(f"[FAISSIndex] 尝试从目录加载索引: {index_dir}")
        if not index_dir.exists():
            logger.warning(f"[FAISSIndex] 索引目录不存在: {index_dir}")
            return
        
        faiss_files = list(index_dir.glob("*.faiss"))
        logger.info(f"[FAISSIndex] 找到 {len(faiss_files)} 个FAISS索引文件")
        
        # 加载所有 .faiss 文件
        for faiss_file in faiss_files:
            sha1 = faiss_file.stem
            try:
                index = faiss.read_index(str(faiss_file))
                self.document_indices[sha1] = index
                logger.info(f"[FAISSIndex] 加载索引文件: {faiss_file.name} (SHA1: {sha1[:16]}..., 向量数: {index.ntotal})")
                
                # 尝试加载对应的 chunk_ids（如果有）
                ids_file = faiss_file.with_suffix('.faiss.ids')
                if ids_file.exists():
                    with open(ids_file, 'rb') as f:
                        chunk_ids = pickle.load(f)
                        self.document_chunk_maps[sha1] = chunk_ids
                    logger.info(f"[FAISSIndex] 加载chunk_ids映射: {ids_file.name} ({len(chunk_ids)} 个chunk_ids)")
                    if len(chunk_ids) > 0:
                        logger.debug(f"[FAISSIndex] 示例chunk_id: {chunk_ids[0]}")
                else:
                    logger.warning(f"[FAISSIndex] 未找到chunk_ids映射文件: {ids_file.name}")
                    logger.warning(f"[FAISSIndex] 这将导致检索时无法将索引位置映射到chunk_id！")
            except Exception as e:
                logger.error(f"[FAISSIndex] 加载索引文件失败 {faiss_file}: {e}", exc_info=True)
        
        logger.info(f"[FAISSIndex] 总共加载 {len(self.document_indices)} 个文档索引")
        logger.info(f"[FAISSIndex] 有chunk_ids映射的文档数: {len(self.document_chunk_maps)}")
    
    def build_index(self, embeddings: List[List[float]], chunk_ids: List[str]):
        """
        构建 FAISS 索引（全局索引模式）
        
        Args:
            embeddings: embedding 向量列表
            chunk_ids: 对应的 chunk_id 列表
        """
        if not embeddings:
            return
            
        dimension = len(embeddings[0])
        # 使用 IndexFlatL2 或 IndexIVFFlat
        self.index = faiss.IndexFlatL2(dimension)
        
        # 转换为 numpy 数组
        np_embeddings = np.array(embeddings).astype('float32')
        self.index.add(np_embeddings)
        self.chunk_ids = chunk_ids
        self.save()
    
    def build_index_for_document(
        self,
        embeddings: List[List[float]],
        chunk_ids: List[str],
        sha1: str,
        use_cosine: bool = True
    ):
        """
        为单个文档构建 FAISS 索引（按文档存储模式）
        
        Args:
            embeddings: embedding 向量列表
            chunk_ids: 对应的 chunk_id 列表
            sha1: 文档的 SHA1 标识
            use_cosine: 是否使用余弦相似度（内积），True 则使用 IndexFlatIP，False 使用 IndexFlatL2
        """
        if not embeddings:
            return
        
        if not self.index_dir:
            raise ValueError("必须指定 index_dir 才能使用按文档存储模式")
        
        dimension = len(embeddings[0])
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        if use_cosine:
            # 使用内积（余弦相似度），需要先归一化
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normalized_embeddings = embeddings_array / norms
            index = faiss.IndexFlatIP(dimension)
            index.add(normalized_embeddings)
        else:
            # 使用 L2 距离
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_array)
        
        # 保存索引文件
        index_dir = Path(self.index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss_file_path = index_dir / f"{sha1}.faiss"
        faiss.write_index(index, str(faiss_file_path))
        
        # 保存 chunk_ids 映射
        ids_file_path = index_dir / f"{sha1}.faiss.ids"
        with open(ids_file_path, 'wb') as f:
            pickle.dump(chunk_ids, f)
        
        # 更新内存中的索引
        self.document_indices[sha1] = index
        self.document_chunk_maps[sha1] = chunk_ids
    
    def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 50,
        document_sha1: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        向量检索
        
        Args:
            query_embedding: 查询向量
            top_k: 返回 Top-K 结果
            document_sha1: 如果指定，则只在该文档的索引中搜索（按文档存储模式）
            
        Returns:
            (chunk_id, similarity) 列表
        """
        # 如果指定了文档 SHA1，使用按文档存储模式
        if document_sha1 and self.index_dir:
            logger.debug(f"[FAISSIndex] 使用文档SHA1限定搜索: {document_sha1[:16]}...")
            if document_sha1 not in self.document_indices:
                logger.warning(f"[FAISSIndex] 文档SHA1不在索引中: {document_sha1[:16]}...")
                return []
            
            index = self.document_indices[document_sha1]
            chunk_ids = self.document_chunk_maps.get(document_sha1, [])
            
            if not chunk_ids:
                logger.error(f"[FAISSIndex] 文档 {document_sha1[:16]}... 没有chunk_ids映射！无法将索引位置映射到chunk_id")
                logger.error(f"[FAISSIndex] 这通常意味着process_chunk_json时没有生成和保存chunk_ids")
            
            logger.debug(f"[FAISSIndex] 文档索引向量数: {index.ntotal}, chunk_ids数量: {len(chunk_ids)}")
            
            np_query = np.array([query_embedding]).astype('float32')
            # 如果使用内积，需要归一化查询向量
            if isinstance(index, faiss.IndexFlatIP):
                norm = np.linalg.norm(np_query)
                if norm > 0:
                    np_query = np_query / norm
            
            distances, indices = index.search(np_query, top_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1:
                    if idx < len(chunk_ids):
                        # 对于内积，值越大越相似；对于 L2，距离越小越相似
                        score = float(distances[0][i])
                        chunk_id = chunk_ids[idx]
                        results.append((chunk_id, score))
                        logger.debug(f"[FAISSIndex] 索引位置 {idx} -> chunk_id: {chunk_id}, score: {score:.4f}")
                    else:
                        logger.warning(f"[FAISSIndex] 索引位置 {idx} 超出chunk_ids范围 (长度: {len(chunk_ids)})")
            
            logger.info(f"[FAISSIndex] 文档限定搜索完成，返回 {len(results)} 个结果")
            return results
        
        # 如果没有指定document_sha1，但有index_dir（按文档存储模式），遍历所有文档索引
        if self.index_dir and self.document_indices:
            logger.info(f"[FAISSIndex] 未指定文档SHA1，遍历所有 {len(self.document_indices)} 个文档索引进行搜索")
            all_results = []
            
            np_query = np.array([query_embedding]).astype('float32')
            
            # 遍历所有文档索引
            for sha1, index in self.document_indices.items():
                chunk_ids = self.document_chunk_maps.get(sha1, [])
                
                if not chunk_ids:
                    logger.warning(f"[FAISSIndex] 文档 {sha1[:16]}... 没有chunk_ids映射，跳过")
                    continue
                
                # 如果使用内积，需要归一化查询向量
                if isinstance(index, faiss.IndexFlatIP):
                    norm = np.linalg.norm(np_query)
                    if norm > 0:
                        np_query_normalized = np_query / norm
                    else:
                        continue
                else:
                    np_query_normalized = np_query
                
                # 在每个文档索引中搜索（取top_k个结果）
                distances, indices = index.search(np_query_normalized, top_k)
                
                for i, idx in enumerate(indices[0]):
                    if idx != -1 and idx < len(chunk_ids):
                        score = float(distances[0][i])
                        chunk_id = chunk_ids[idx]
                        all_results.append((chunk_id, score))
            
            # 对所有结果按分数排序，取top_k
            # 对于L2距离，距离越小越好；对于内积，值越大越好
            if all_results:
                # 判断第一个索引的类型来决定排序方式
                first_index = next(iter(self.document_indices.values()))
                if isinstance(first_index, faiss.IndexFlatIP):
                    # 内积：值越大越好，降序
                    all_results.sort(key=lambda x: x[1], reverse=True)
                else:
                    # L2距离：距离越小越好，升序
                    all_results.sort(key=lambda x: x[1])
                
                final_results = all_results[:top_k]
                logger.info(f"[FAISSIndex] 遍历所有文档索引完成，从 {len(all_results)} 个候选结果中选择 top {len(final_results)} 个")
                return final_results
            else:
                logger.warning(f"[FAISSIndex] 遍历所有文档索引，未找到任何结果")
                return []
        
        # 全局索引模式（如果存在）
        if self.index is None:
            logger.warning(f"[FAISSIndex] 全局索引未初始化，且没有可用的文档索引")
            return []
        
        logger.debug(f"[FAISSIndex] 使用全局索引模式，向量数: {self.index.ntotal}, chunk_ids数量: {len(self.chunk_ids)}")
            
        np_query = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(np_query, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                if idx < len(self.chunk_ids):
                    # FAISS 返回的是距离（L2 距离越小越相似）
                    chunk_id = self.chunk_ids[idx]
                    score = float(distances[0][i])
                    results.append((chunk_id, score))
                    logger.debug(f"[FAISSIndex] 索引位置 {idx} -> chunk_id: {chunk_id}, distance: {score:.4f}")
                else:
                    logger.warning(f"[FAISSIndex] 索引位置 {idx} 超出chunk_ids范围 (长度: {len(self.chunk_ids)})")
        
        logger.info(f"[FAISSIndex] 全局索引搜索完成，返回 {len(results)} 个结果")
        return results
    
    def save(self):
        """保存索引到文件"""
        if self.index is not None:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            faiss.write_index(self.index, self.index_path)
            with open(self.id_map_path, 'wb') as f:
                pickle.dump(self.chunk_ids, f)
    
    def load(self):
        """从文件加载索引"""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            if os.path.exists(self.id_map_path):
                with open(self.id_map_path, 'rb') as f:
                    self.chunk_ids = pickle.load(f)

