"""
向量数据库服务
为每个文档单独创建和保存 FAISS 索引
参考 RAG-cy 的 VectorDBIngestor 实现
"""
import json
import logging
import uuid
import faiss
import numpy as np
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from app.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)


class VectorDBService:
    """向量数据库服务类，用于创建和管理 FAISS 索引"""
    
    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        """
        初始化向量数据库服务
        
        Args:
            embedding_service: Embedding 服务实例，如果为 None 则自动创建
        """
        self.embedding_service = embedding_service or EmbeddingService()
    
    def _create_vector_index(self, embeddings: List[List[float]]) -> faiss.Index:
        """
        使用 FAISS 构建向量索引，采用内积（余弦距离）
        
        Args:
            embeddings: embedding 向量列表
            
        Returns:
            FAISS 索引对象
        """
        if not embeddings:
            raise ValueError("embeddings 列表不能为空")
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dimension = len(embeddings[0])
        
        # 使用 IndexFlatIP（内积）进行余弦相似度计算
        # 注意：使用内积前需要先对向量进行归一化
        # 归一化向量
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        norms[norms == 0] = 1  # 避免除零
        normalized_embeddings = embeddings_array / norms
        
        index = faiss.IndexFlatIP(dimension)
        index.add(normalized_embeddings)
        
        return index
    
    async def process_chunk_json(
        self,
        chunk_json_path: str,
        output_dir: str,
        max_chunk_length: int = 2048
    ) -> str:
        """
        处理单个 chunk JSON 文件，生成并保存 FAISS 索引
        
        Args:
            chunk_json_path: chunk JSON 文件路径
            output_dir: 输出目录
            max_chunk_length: 最大 chunk 长度（字符数），超长内容会被截断
            
        Returns:
            FAISS 索引文件路径
        """
        chunk_json_path = Path(chunk_json_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载 chunk JSON
        with open(chunk_json_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        
        # 提取 metainfo 和 chunks
        metainfo = report_data.get("metainfo", {})
        chunks = report_data.get("content", {}).get("chunks", [])
        
        logger.info(f"[VectorDB] 处理chunk JSON文件: {chunk_json_path.name}")
        logger.info(f"[VectorDB] chunks数量: {len(chunks)}")
        
        # 获取 SHA1 作为文件名
        sha1 = metainfo.get("sha1", "")
        file_name = metainfo.get("file_name", "")
        if not sha1:
            raise ValueError(f"chunk JSON 文件 {chunk_json_path} 缺少 sha1 字段")
        
        logger.info(f"[VectorDB] 文档SHA1: {sha1}, 文件名: {file_name}")
        
        # 提取文本块并生成chunk_ids
        text_chunks = []
        chunk_ids = []
        document_name = Path(file_name).stem if file_name else Path(chunk_json_path).stem
        
        for idx, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            if text and len(text) > 0:
                # 截断超长内容
                if len(text) > max_chunk_length:
                    text = text[:max_chunk_length]
                text_chunks.append(text)
                
                # 生成chunk_id（如果chunk中没有chunk_id字段）
                chunk_id = chunk.get("chunk_id")
                if not chunk_id:
                    # 使用格式: {sha1}_{index} 作为chunk_id
                    chunk_id = f"{sha1}_{idx}"
                chunk_ids.append(chunk_id)
        
        logger.info(f"[VectorDB] 提取了 {len(text_chunks)} 个有效文本块")
        logger.info(f"[VectorDB] 生成了 {len(chunk_ids)} 个chunk_ids")
        if len(chunk_ids) > 0:
            logger.debug(f"[VectorDB] 示例chunk_id: {chunk_ids[0]}")
        
        if not text_chunks:
            raise ValueError(f"chunk JSON 文件 {chunk_json_path} 中没有有效的文本块")
        
        if len(text_chunks) != len(chunk_ids):
            logger.error(f"[VectorDB] 文本块数量({len(text_chunks)})与chunk_ids数量({len(chunk_ids)})不匹配！")
        
        # 生成 embeddings（异步）
        logger.info(f"[VectorDB] 开始生成embeddings...")
        embeddings = await self.embedding_service.embed_documents(text_chunks)
        logger.info(f"[VectorDB] 生成了 {len(embeddings)} 个embeddings (维度: {len(embeddings[0]) if embeddings else 0})")
        
        # 创建 FAISS 索引
        logger.info(f"[VectorDB] 创建FAISS索引...")
        index = self._create_vector_index(embeddings)
        logger.info(f"[VectorDB] FAISS索引创建完成，向量数: {index.ntotal}")
        
        # 保存索引文件
        faiss_file_path = output_dir / f"{sha1}.faiss"
        faiss.write_index(index, str(faiss_file_path))
        logger.info(f"[VectorDB] FAISS索引已保存: {faiss_file_path}")
        
        # 保存 chunk_ids 映射文件
        ids_file_path = output_dir / f"{sha1}.faiss.ids"
        import pickle
        with open(ids_file_path, 'wb') as f:
            pickle.dump(chunk_ids, f)
        logger.info(f"[VectorDB] chunk_ids映射已保存: {ids_file_path} ({len(chunk_ids)} 个chunk_ids)")
        
        # 注意：这里没有将chunks加载到MetadataStorage
        # 需要额外的步骤将chunks从JSON加载到MetadataStorage.chunks字典中
        logger.warning(f"[VectorDB] 注意：chunks尚未加载到MetadataStorage，需要额外步骤！")
        logger.warning(f"[VectorDB] 这可能导致检索时无法从MetadataStorage获取chunk详情")
        
        return str(faiss_file_path)
    
    async def process_chunk_json_directory(
        self,
        chunk_json_dir: str,
        output_dir: str,
        max_chunk_length: int = 2048
    ) -> List[str]:
        """
        批量处理目录下的所有 chunk JSON 文件
        
        Args:
            chunk_json_dir: chunk JSON 文件目录
            output_dir: 输出目录
            max_chunk_length: 最大 chunk 长度
            
        Returns:
            FAISS 索引文件路径列表
        """
        chunk_json_dir = Path(chunk_json_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_json_paths = list(chunk_json_dir.glob("*.json"))
        faiss_files = []
        
        for json_path in tqdm(all_json_paths, desc="Processing chunk JSON files for FAISS"):
            try:
                faiss_path = await self.process_chunk_json(
                    str(json_path),
                    str(output_dir),
                    max_chunk_length=max_chunk_length
                )
                faiss_files.append(faiss_path)
                print(f"已处理: {json_path.name} -> {Path(faiss_path).name}")
            except Exception as e:
                print(f"处理失败 {json_path.name}: {e}")
        
        print(f"共处理 {len(faiss_files)} 个文件")
        return faiss_files
