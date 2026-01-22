"""
Embedding 服务
使用 Qwen Embedding v3 生成向量
"""
import os
import asyncio
from typing import List
import dashscope
from dashscope import TextEmbedding
from dotenv import load_dotenv

load_dotenv()

class EmbeddingService:
    """Embedding 服务类，使用 DashScope API 调用通义千问 Embedding 模型（支持异步）"""
    
    def __init__(self):
        self.api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("未找到 QWEN_API_KEY 或 DASHSCOPE_API_KEY 环境变量")
        dashscope.api_key = self.api_key
        self.model = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
    
    def _embed_sync(self, texts: List[str]) -> List[List[float]]:
        """
        同步调用 DashScope API 生成 embedding
        
        Args:
            texts: 文本列表
            
        Returns:
            embedding 向量列表
        """
        if not texts:
            return []
        
        # 过滤空文本
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return []
        
        embeddings = []
        MAX_BATCH_SIZE = 10  # DashScope API 批量限制（实际限制为10）
        
        # 分批处理
        for i in range(0, len(valid_texts), MAX_BATCH_SIZE):
            batch = valid_texts[i:i+MAX_BATCH_SIZE]
            
            try:
                resp = TextEmbedding.call(
                    model=TextEmbedding.Models.text_embedding_v3,
                    input=batch
                )
                
                # 检查响应是否成功
                if resp is None:
                    raise RuntimeError("DashScope API返回None")
                
                # 检查状态码
                if hasattr(resp, 'status_code') and resp.status_code != 200:
                    error_msg = getattr(resp, 'message', 'Unknown error')
                    raise RuntimeError(f"DashScope API调用失败: status_code={resp.status_code}, message={error_msg}")
                
                # 处理响应
                if 'output' in resp and resp['output'] and 'embeddings' in resp['output']:
                    for emb in resp['output']['embeddings']:
                        if emb.get('embedding') and len(emb['embedding']) > 0:
                            embeddings.append(emb['embedding'])
                        else:
                            raise RuntimeError(f"DashScope返回的embedding为空，text_index={emb.get('text_index', None)}")
                elif 'output' in resp and resp['output'] and 'embedding' in resp['output']:
                    # 单条输入的情况
                    if resp['output']['embedding'] and len(resp['output']['embedding']) > 0:
                        embeddings.append(resp['output']['embedding'])
                    else:
                        raise RuntimeError("DashScope返回的embedding为空")
                else:
                    raise RuntimeError(f"DashScope embedding API返回格式异常: {resp}")
                    
            except Exception as e:
                raise Exception(f"生成 Embedding 失败: {str(e)}")
        
        return embeddings
    
    async def embed_query(self, query: str) -> List[float]:
        """
        对查询文本生成 embedding (异步)
        
        Args:
            query: 查询文本
            
        Returns:
            embedding 向量
        """
        # 使用线程池执行同步调用
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, 
            self._embed_sync, 
            [query]
        )
        
        if not embeddings:
            raise Exception("生成 Query Embedding 失败：返回结果为空")
        
        return embeddings[0]
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量对文档生成 embedding (异步)
        
        Args:
            texts: 文本列表
            
        Returns:
            embedding 向量列表
        """
        # 使用线程池执行同步调用
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            self._embed_sync,
            texts
        )
        
        if not embeddings:
            raise Exception("生成 Document Embedding 失败：返回结果为空")
        
        return embeddings

