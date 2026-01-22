"""
重排服务
使用 Jina Reranker API 对检索结果进行重排
"""
from typing import List, Dict, Optional
import os
import requests
import logging

logger = logging.getLogger(__name__)


class RerankService:
    """重排服务类"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        初始化 Jina Reranker 服务
        
        Args:
            api_key: Jina API 密钥（从环境变量 JINA_API_KEY 读取，或直接传入）
            base_url: Jina API 基础 URL（可选，默认使用官方 API）
        """
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        
        self.base_url = base_url or os.getenv(
            "JINA_API_BASE_URL", 
            "https://api.jina.ai/v1/rerank"
        )
        self.model = os.getenv("JINA_RERANK_MODEL", "jina-reranker-v2-base-multilingual")
    
    def rerank(
        self, 
        query: str, 
        documents: List[Dict], 
        top_k: int = 10
    ) -> List[Dict]:
        """
        对检索结果进行重排
        
        Args:
            query: 查询文本
            documents: 检索结果列表（包含 chunk_id, text, similarity 等）
            top_k: 返回 Top-K 结果
            
        Returns:
            重排后的结果列表，按相关性从高到低排序
        """
        if not documents:
            return []
        
        if not self.api_key:
            logger.warning("JINA_API_KEY 未设置，跳过重排流程")
            return sorted(documents, key=lambda x: x.get("similarity", 0.0), reverse=True)[:top_k]
        texts = [doc.get("text", "") for doc in documents]
        
        # 调用 Jina Reranker API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "query": query,
            "documents": texts,
            "top_n": top_k
        }
        
        try:
            response = requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            # 解析返回结果
            reranked_results = []
            if "results" in result:
                for item in result["results"]:
                    # 获取原始文档索引
                    doc_index = item.get("index", 0)
                    if doc_index < len(documents):
                        original_doc = documents[doc_index].copy()
                        # 更新相似度分数
                        original_doc["similarity"] = item.get("relevance_score", 0.0)
                        original_doc["rerank_score"] = item.get("relevance_score", 0.0)
                        reranked_results.append(original_doc)
            
            return reranked_results
            
        except requests.exceptions.RequestException as e:
            # 如果 API 调用失败，返回原始结果（按原始相似度排序）
            logger.error(f"Jina Reranker API 调用失败: {e}")
            sorted_docs = sorted(
                documents, 
                key=lambda x: x.get("similarity", 0.0), 
                reverse=True
            )
            return sorted_docs[:top_k]
        except Exception as e:
            # 处理其他异常
            logger.error(f"Jina Reranker 处理异常: {e}")
            sorted_docs = sorted(
                documents, 
                key=lambda x: x.get("similarity", 0.0), 
                reverse=True
            )
            return sorted_docs[:top_k]

