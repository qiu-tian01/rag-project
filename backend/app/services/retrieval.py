"""
检索服务
整合向量检索、重排等流程
"""
import logging
import time
import os
from typing import List, Dict, Tuple, Optional, Set
from rank_bm25 import BM25Okapi
import jieba
from app.services.embedding import EmbeddingService
from app.services.rerank import RerankService
from app.storage.faiss_index import FAISSIndex
from app.storage.metadata import MetadataStorage

logger = logging.getLogger(__name__)

class RetrievalService:
    """检索服务类，整合向量检索和关键字检索（BM25）"""
    
    def __init__(self):
        logger.info("[Retrieval] 初始化RetrievalService...")
        self.embedding_service = EmbeddingService()
        self.rerank_service = RerankService()
        
        # 初始化FAISS索引（需要设置index_dir以支持按文档存储模式）
        index_dir = os.getenv("FAISS_INDEX_DIR", "./data/metadata/vector_dbs")
        logger.info(f"[Retrieval] FAISS索引目录: {index_dir}")
        self.faiss_index = FAISSIndex(index_dir=index_dir)
        
        # 使用默认路径初始化MetadataStorage
        chunked_reports_dir = os.getenv("CHUNKED_REPORTS_DIR", "./data/metadata/chunked_reports")
        logger.info(f"[Retrieval] Chunked reports目录: {chunked_reports_dir}")
        self.metadata_storage = MetadataStorage(
            chunked_reports_dir=chunked_reports_dir
        )
        
        self.bm25 = None
        self.bm25_chunk_ids = []
        self._init_bm25()
        logger.info("[Retrieval] RetrievalService初始化完成")
    
    def _init_bm25(self):
        """初始化 BM25 索引"""
        chunks = list(self.metadata_storage.chunks.values())
        logger.info(f"[Retrieval] 初始化BM25索引，MetadataStorage中有 {len(chunks)} 个chunks")
        if not chunks:
            logger.warning(f"[Retrieval] MetadataStorage为空，无法初始化BM25索引")
            logger.warning(f"[Retrieval] 这通常意味着chunks.json文件不存在或为空，或者chunk JSON文件未被加载")
            return
            
        # 使用 jieba 对中文进行分词
        logger.info(f"[Retrieval] 开始对 {len(chunks)} 个chunks进行jieba分词...")
        tokenized_corpus = [list(jieba.cut(chunk['text'])) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.bm25_chunk_ids = [chunk['chunk_id'] for chunk in chunks]
        logger.info(f"[Retrieval] BM25索引初始化完成，chunk_ids数量: {len(self.bm25_chunk_ids)}")
        if len(self.bm25_chunk_ids) > 0:
            logger.debug(f"[Retrieval] 示例chunk_id: {self.bm25_chunk_ids[0]}")
    
    def _filter_by_document_name(self, product_name: str, chunk_ids: Optional[List[str]] = None) -> Set[str]:
        """
        根据产品名称/文档名称过滤chunks
        
        Args:
            product_name: 产品名称或文档名称
            chunk_ids: 可选的chunk ID列表，如果提供则只在这些chunks中过滤
            
        Returns:
            匹配的chunk ID集合
        """
        # 获取匹配的chunk IDs
        matched_chunk_ids = set(self.metadata_storage.get_chunk_ids_by_document_name(product_name, fuzzy_match=True))
        
        # 如果提供了chunk_ids，取交集
        if chunk_ids:
            matched_chunk_ids = matched_chunk_ids.intersection(set(chunk_ids))
        
        return matched_chunk_ids
    
    async def search(
        self, 
        query: str, 
        top_k: int = 10,
        search_mode: int = 2,
        product_name: Optional[str] = None,
        document_sha1: Optional[str] = None
    ) -> List[Dict]:
        """
        检索流程 (异步)
        
        Args:
            query: 查询文本
            top_k: 返回Top-K结果
            search_mode: 搜索模式，1=纯向量搜索，2=混合检索+rerank
            product_name: 可选的产品名称，用于过滤相关文档
            document_sha1: 可选的文档SHA1，用于限定搜索范围
        """
        search_start = time.time()
        logger.info(f"[Retrieval] 开始检索流程 (模式={search_mode}, top_k={top_k})")
        
        # 步骤1: 如果提供产品名称，先过滤相关文档的chunks
        step_start = time.time()
        filtered_chunk_ids = None
        if product_name:
            filtered_chunk_ids = list(self._filter_by_document_name(product_name))
            logger.info(f"[Retrieval] 产品名称过滤: '{product_name}' -> {len(filtered_chunk_ids)} 个chunks (耗时: {time.time() - step_start:.2f}秒)")
            if not filtered_chunk_ids:
                logger.warning(f"[Retrieval] 未找到匹配的文档，回退到全库搜索")
        else:
            logger.debug(f"[Retrieval] 未提供产品名称，使用全库搜索")
        
        # 步骤2: 向量检索
        step_start = time.time()
        query_embedding = await self.embedding_service.embed_query(query)
        logger.debug(f"[Retrieval] 生成查询向量完成 (维度: {len(query_embedding)})")
        
        # 如果指定了document_sha1，使用按文档存储模式
        if document_sha1:
            logger.info(f"[Retrieval] 使用文档SHA1限定搜索范围: {document_sha1[:16]}...")
            vector_results = self.faiss_index.search(query_embedding, top_k=50, document_sha1=document_sha1)
        else:
            vector_results = self.faiss_index.search(query_embedding, top_k=50)
        
        logger.info(f"[Retrieval] 向量检索完成，获得 {len(vector_results)} 个候选结果 (耗时: {time.time() - step_start:.2f}秒)")
        if vector_results:
            logger.debug(f"[Retrieval] 向量检索Top1相似度: {vector_results[0][1]:.4f}")
        
        # 如果提供了产品名称过滤，过滤向量检索结果
        if filtered_chunk_ids:
            before_count = len(vector_results)
            vector_results = [(cid, score) for cid, score in vector_results if cid in filtered_chunk_ids]
            logger.info(f"[Retrieval] 产品名称过滤后: {before_count} -> {len(vector_results)} 个结果")
        
        # 步骤3: 根据search_mode选择检索策略
        if search_mode == 1:
            # 纯向量搜索模式：只使用向量检索，跳过BM25和rerank
            logger.info(f"[Retrieval] 使用纯向量搜索模式")
            chunk_details = []
            missing_chunks = []
            for chunk_id, score in vector_results[:top_k]:
                detail = self.metadata_storage.get_chunk(chunk_id)
                if detail:
                    # 对于L2距离，距离越小越相似，转换为相似度分数
                    # 这里假设score是距离，需要转换为相似度
                    if isinstance(score, float) and score > 0:
                        # L2距离转相似度：相似度 = 1 / (1 + 距离)
                        similarity = 1.0 / (1.0 + score)
                    else:
                        similarity = float(score) if score > 0 else 0.0
                    detail['similarity'] = similarity
                    chunk_details.append(detail)
                    logger.debug(f"[Retrieval] 成功获取chunk详情: {chunk_id[:20]}... (文档: {detail.get('document_name', 'N/A')})")
                else:
                    missing_chunks.append(chunk_id)
                    logger.warning(f"[Retrieval] 无法从MetadataStorage获取chunk详情: {chunk_id}")
            
            if missing_chunks:
                logger.error(f"[Retrieval] 有 {len(missing_chunks)} 个chunk无法获取详情！")
                logger.error(f"[Retrieval] MetadataStorage中总共有 {len(self.metadata_storage.chunks)} 个chunks")
            
            total_time = time.time() - search_start
            logger.info(f"[Retrieval] 检索完成，返回 {len(chunk_details)} 个结果 (总耗时: {total_time:.2f}秒)")
            return chunk_details
        else:
            # 混合检索模式：向量 + BM25 + rerank
            logger.info(f"[Retrieval] 使用混合检索模式 (向量 + BM25 + rerank)")
            
            # 2. BM25 检索
            step_start = time.time()
            bm25_results = []
            if self.bm25:
                tokenized_query = list(jieba.cut(query))
                scores = self.bm25.get_scores(tokenized_query)
                # 获取前 50 个结果
                top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:50]
                bm25_results = [(self.bm25_chunk_ids[i], float(scores[i])) for i in top_n_indices if scores[i] > 0]
                logger.info(f"[Retrieval] BM25检索完成，获得 {len(bm25_results)} 个结果 (耗时: {time.time() - step_start:.2f}秒)")
                
                # 如果提供了产品名称过滤，过滤BM25结果
                if filtered_chunk_ids:
                    before_count = len(bm25_results)
                    bm25_results = [(cid, score) for cid, score in bm25_results if cid in filtered_chunk_ids]
                    logger.debug(f"[Retrieval] BM25过滤后: {before_count} -> {len(bm25_results)} 个结果")
            else:
                logger.warning(f"[Retrieval] BM25索引未初始化，跳过BM25检索")
            
            # 3. 合并结果
            step_start = time.time()
            combined_results = self._combine_results(vector_results, bm25_results)
            logger.info(f"[Retrieval] 结果合并完成，获得 {len(combined_results)} 个合并结果 (耗时: {time.time() - step_start:.2f}秒)")
            
            # 4. 获取详细元数据
            step_start = time.time()
            chunk_details = []
            missing_chunks = []
            for chunk_id, score in combined_results[:20]:
                detail = self.metadata_storage.get_chunk(chunk_id)
                if detail:
                    detail['similarity'] = score
                    chunk_details.append(detail)
                    logger.debug(f"[Retrieval] 成功获取chunk详情: {chunk_id[:20]}... (文档: {detail.get('document_name', 'N/A')})")
                else:
                    missing_chunks.append(chunk_id)
                    logger.warning(f"[Retrieval] 无法从MetadataStorage获取chunk详情: {chunk_id}")
                    logger.warning(f"[Retrieval] 这通常意味着chunk JSON文件中的chunks没有被加载到MetadataStorage")
            
            if missing_chunks:
                logger.error(f"[Retrieval] 有 {len(missing_chunks)} 个chunk无法获取详情！")
                logger.error(f"[Retrieval] 示例缺失的chunk_id: {missing_chunks[0] if missing_chunks else 'N/A'}")
                logger.error(f"[Retrieval] MetadataStorage中总共有 {len(self.metadata_storage.chunks)} 个chunks")
            
            logger.info(f"[Retrieval] 获取元数据完成，获得 {len(chunk_details)} 个详细结果 (耗时: {time.time() - step_start:.2f}秒)")
            if len(chunk_details) == 0 and len(combined_results) > 0:
                logger.error(f"[Retrieval] 严重问题：检索到 {len(combined_results)} 个结果，但无法获取任何chunk详情！")
                logger.error(f"[Retrieval] 请检查：1) MetadataStorage是否加载了chunks 2) chunk_id映射是否正确")
                    
            # 5. 重排 (Jina API 目前是同步调用，或可改为异步)
            if chunk_details and self.rerank_service:
                step_start = time.time()
                logger.info(f"[Retrieval] 开始重排，候选数量: {len(chunk_details)}")
                reranked_results = self.rerank_service.rerank(query, chunk_details)
                logger.info(f"[Retrieval] 重排完成，返回 {len(reranked_results[:top_k])} 个结果 (耗时: {time.time() - step_start:.2f}秒)")
                
                total_time = time.time() - search_start
                logger.info(f"[Retrieval] 混合检索完成，总耗时: {total_time:.2f}秒")
                return reranked_results[:top_k]
            
            total_time = time.time() - search_start
            logger.info(f"[Retrieval] 混合检索完成 (未重排)，返回 {len(chunk_details[:top_k])} 个结果，总耗时: {total_time:.2f}秒")
            return chunk_details[:top_k]

    def _combine_results(self, vector_results: List[Tuple[str, float]], bm25_results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """合并向量检索和 BM25 结果"""
        # 简单权重融合
        scores = {}
        for cid, dist in vector_results:
            # 距离越小分越高，这里简单处理
            # 对于L2距离，转换为相似度分数
            if isinstance(dist, float) and dist > 0:
                similarity = 1.0 / (1.0 + dist)
            else:
                similarity = float(dist) if dist > 0 else 0.0
            scores[cid] = scores.get(cid, 0) + similarity * 0.7
            
        for cid, score in bm25_results:
            scores[cid] = scores.get(cid, 0) + (score / 10.0) * 0.3
            
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

