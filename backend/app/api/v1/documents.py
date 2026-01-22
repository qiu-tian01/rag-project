"""
文档处理 API
提供 PDF 文件上传和处理接口
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
import os

from app.services.document_processor import DocumentProcessor
from app.services.pipeline import PipelinePaths

router = APIRouter(prefix="/api/v1", tags=["documents"])

# 使用单例
_processor = None

def get_processor():
    """获取文档处理服务实例"""
    global _processor
    if _processor is None:
        _processor = DocumentProcessor()
    return _processor


class ProcessResponse(BaseModel):
    """文档处理响应模型"""
    success: bool
    message: str
    file_name: Optional[str] = None
    sha1: Optional[str] = None
    markdown_path: Optional[str] = None
    chunk_json_path: Optional[str] = None
    faiss_path: Optional[str] = None
    chunk_count: Optional[int] = None
    error: Optional[str] = None


class ProcessStatusResponse(BaseModel):
    """处理状态响应模型"""
    status: str  # "processing", "completed", "failed"
    progress: float  # 0.0 - 1.0
    message: str
    result: Optional[ProcessResponse] = None


@router.post("/documents/upload", response_model=ProcessResponse)
async def upload_and_process_document(
    file: UploadFile = File(..., description="PDF 文件"),
    company_name: Optional[str] = Form(None, description="公司名称（可选）"),
    chunk_size: int = Form(30, description="每个 chunk 的最大行数"),
    chunk_overlap: int = Form(5, description="chunk 之间的重叠行数"),
    processor: DocumentProcessor = Depends(get_processor)
):
    """
    上传并处理 PDF 文档
    
    完整流程：
    1. 接收上传的 PDF 文件
    2. PDF → Markdown 转换
    3. Markdown → Chunks 切分
    4. Chunks → Embeddings 生成
    5. 创建并保存 FAISS 向量索引
    
    返回处理结果，包括生成的文件路径和统计信息。
    """
    # 验证文件类型
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="只支持 PDF 文件格式"
        )
    
    # 读取文件内容
    try:
        file_content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"读取文件失败: {str(e)}"
        )
    
    if len(file_content) == 0:
        raise HTTPException(
            status_code=400,
            detail="文件内容为空"
        )
    
    # 处理文件
    result = await processor.process_uploaded_file(
        file_content=file_content,
        file_name=file.filename,
        company_name=company_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    if not result["success"]:
        raise HTTPException(
            status_code=500,
            detail=result.get("message", "处理失败")
        )
    
    return ProcessResponse(**result)


@router.post("/documents/process", response_model=ProcessResponse)
async def process_existing_document(
    file_path: str = Form(..., description="PDF 文件路径（相对于 documents 目录）"),
    company_name: Optional[str] = Form(None, description="公司名称（可选）"),
    chunk_size: int = Form(30, description="每个 chunk 的最大行数"),
    chunk_overlap: int = Form(5, description="chunk 之间的重叠行数"),
    processor: DocumentProcessor = Depends(get_processor)
):
    """
    处理已存在的 PDF 文档
    
    用于处理已经上传到服务器的 PDF 文件。
    文件路径应该是相对于 documents 目录的路径。
    """
    # 构建完整路径
    full_path = processor.paths.documents_dir / file_path
    
    if not full_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"文件不存在: {file_path}"
        )
    
    # 处理文件
    result = await processor.process_pdf_file(
        pdf_file_path=str(full_path),
        company_name=company_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    if not result["success"]:
        raise HTTPException(
            status_code=500,
            detail=result.get("message", "处理失败")
        )
    
    return ProcessResponse(**result)


@router.get("/documents/list")
async def list_documents(
    processor: DocumentProcessor = Depends(get_processor)
):
    """
    列出所有已处理的文档
    
    返回已处理的文档列表，包括文件名、SHA1、处理状态等信息。
    """
    documents_dir = processor.paths.documents_dir
    chunked_reports_dir = processor.paths.chunked_reports_dir
    vector_dbs_dir = processor.paths.vector_dbs_dir
    
    documents = []
    
    # 遍历 documents 目录
    for pdf_file in documents_dir.glob("*.pdf"):
        pdf_sha1 = None
        try:
            from app.utils.hash_utils import calculate_file_sha1
            pdf_sha1 = calculate_file_sha1(pdf_file)
        except:
            pass
        
        # 检查是否已处理
        chunk_json = chunked_reports_dir / f"{pdf_file.stem}.json"
        faiss_file = vector_dbs_dir / f"{pdf_sha1}.faiss" if pdf_sha1 else None
        
        processed = chunk_json.exists() and (faiss_file.exists() if faiss_file else False)
        
        documents.append({
            "file_name": pdf_file.name,
            "file_path": str(pdf_file.relative_to(documents_dir)),
            "file_size": pdf_file.stat().st_size,
            "sha1": pdf_sha1,
            "processed": processed,
            "chunk_json_exists": chunk_json.exists(),
            "faiss_exists": faiss_file.exists() if faiss_file else False
        })
    
    return {
        "total": len(documents),
        "documents": documents
    }
