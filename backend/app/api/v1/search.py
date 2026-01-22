"""
检索 API
"""
import logging
import time
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict
from app.services.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["search"])

# 使用单例或依赖注入
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10


class SearchResult(BaseModel):
    chunk_id: str
    document_name: str
    section_path: List[str]
    text: str
    similarity: float
    page_num: Optional[int] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total: int


class ChatRequest(BaseModel):
    query: str
    history: Optional[List[Dict[str, str]]] = []
    stream: Optional[bool] = False
    search_mode: Optional[int] = 2  # 1=纯向量搜索, 2=混合检索+rerank
    llm_model: Optional[int] = 2  # 1=qwen-max, 2=qwen-plus, 3=qwen-turbo
    product_name: Optional[str] = None  # 可选的产品名称过滤


class ChatResponse(BaseModel):
    answer: str
    thoughts: Optional[str] = None
    citations: Optional[List[int]] = []
    sources: List[SearchResult]


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, pipeline: RAGPipeline = Depends(get_pipeline)):
    """
    检索接口：根据查询文本返回 Top-K 检索结果
    """
    results = await pipeline.retrieval_service.search(request.query, top_k=request.top_k)
    return SearchResponse(results=results, total=len(results))


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, pipeline: RAGPipeline = Depends(get_pipeline)):
    """
    问答接口：基于 RAG 的增强问答
    
    参数说明：
    - search_mode: 1=纯向量搜索, 2=混合检索+rerank（默认2）
    - llm_model: 1=qwen-max, 2=qwen-plus, 3=qwen-turbo（默认2）
    - product_name: 可选的产品名称，用于过滤相关文档
    """
    start_time = time.time()
    
    logger.info("=" * 80)
    logger.info("收到 Chat 请求")
    logger.info(f"查询内容: {request.query[:100]}..." if len(request.query) > 100 else f"查询内容: {request.query}")
    logger.info(f"搜索模式: {request.search_mode} ({'纯向量搜索' if request.search_mode == 1 else '混合检索+rerank'})")
    logger.info(f"大模型: {request.llm_model} ({'qwen-max' if request.llm_model == 1 else 'qwen-plus' if request.llm_model == 2 else 'qwen-turbo'})")
    logger.info(f"产品名称: {request.product_name or '未指定'}")
    logger.info(f"历史记录数: {len(request.history) if request.history else 0}")
    
    try:
        result = await pipeline.answer(
            query=request.query,
            history=request.history,
            search_mode=request.search_mode,
            llm_model=request.llm_model,
            product_name=request.product_name
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"请求处理完成，耗时: {elapsed_time:.2f}秒")
        logger.info(f"返回答案长度: {len(result.get('answer', ''))} 字符")
        logger.info(f"引用页码数: {len(result.get('citations', []))}")
        logger.info(f"来源文档数: {len(result.get('sources', []))}")
        logger.info("=" * 80)
        
        return ChatResponse(
            answer=result["answer"],
            thoughts=result.get("thoughts"),
            citations=result.get("citations"),
            sources=result["sources"]
        )
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"处理请求时发生错误，耗时: {elapsed_time:.2f}秒")
        logger.error(f"错误信息: {str(e)}", exc_info=True)
        logger.info("=" * 80)
        raise

