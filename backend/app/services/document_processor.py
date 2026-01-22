"""
文档处理服务
处理单个 PDF 文件的完整流程：转换 → Chunk → Embedding → 存储向量数据库
"""
from typing import Dict, Optional
from pathlib import Path
import json
import tempfile
import shutil

from app.services.pdf_to_markdown import PDFToMarkdownService
from app.services.chunking import DocumentChunker
from app.services.vector_db import VectorDBService
from app.services.embedding import EmbeddingService
from app.utils.hash_utils import calculate_file_sha1
from app.services.pipeline import PipelinePaths


class DocumentProcessor:
    """文档处理服务类，用于处理单个 PDF 文件"""
    
    def __init__(self, paths: Optional[PipelinePaths] = None):
        """
        初始化文档处理服务
        
        Args:
            paths: 路径配置对象，如果为 None 则使用默认路径
        """
        self.paths = paths or PipelinePaths()
        self.pdf_to_markdown = PDFToMarkdownService()
        self.chunker = DocumentChunker()
        self.embedding_service = EmbeddingService()
        self.vector_db = VectorDBService(self.embedding_service)
    
    async def process_pdf_file(
        self,
        pdf_file_path: str,
        company_name: Optional[str] = None,
        chunk_size: int = 30,
        chunk_overlap: int = 5
    ) -> Dict:
        """
        处理单个 PDF 文件的完整流程
        
        流程：
        1. PDF → Markdown（保存到 debug_data）
        2. Markdown → Chunks（保存到 metadata/chunked_reports）
        3. Chunks → Embeddings → FAISS（保存到 metadata/vector_dbs）
        
        Args:
            pdf_file_path: PDF 文件路径
            company_name: 公司名称（可选，用于元数据）
            chunk_size: 每个 chunk 的最大行数（默认 30）
            chunk_overlap: chunk 之间的重叠行数（默认 5）
            
        Returns:
            处理结果字典，包含：
            - success: 是否成功
            - message: 处理消息
            - file_name: 文件名
            - sha1: SHA1 哈希值
            - markdown_path: Markdown 文件路径
            - chunk_json_path: Chunk JSON 文件路径
            - faiss_path: FAISS 索引文件路径
            - chunk_count: Chunk 数量
            - error: 错误信息（如果失败）
        """
        pdf_path = Path(pdf_file_path)
        
        if not pdf_path.exists():
            return {
                "success": False,
                "message": f"PDF 文件不存在: {pdf_file_path}",
                "error": "FileNotFoundError"
            }
        
        try:
            # 计算 PDF 的 SHA1
            pdf_sha1 = calculate_file_sha1(pdf_path)
            
            # 步骤 1: PDF → Markdown
            md_path = self.pdf_to_markdown.convert_pdf_to_markdown(
                str(pdf_path),
                str(self.paths.markdown_dir)
            )
            
            # 步骤 2: Markdown → Chunks (保存为 JSON)
            chunk_json_path = self.paths.chunked_reports_dir / f"{pdf_path.stem}.json"
            self.chunker.chunk_markdown_and_save(
                md_path=str(md_path),
                output_path=str(chunk_json_path),
                sha1=pdf_sha1,
                company_name=company_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # 读取 chunk 数量
            with open(chunk_json_path, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            chunk_count = len(chunk_data['content']['chunks'])
            
            # 步骤 3: Chunks → Embeddings → FAISS
            faiss_path = await self.vector_db.process_chunk_json(
                chunk_json_path=str(chunk_json_path),
                output_dir=str(self.paths.vector_dbs_dir)
            )
            
            return {
                "success": True,
                "message": "文档处理成功",
                "file_name": pdf_path.name,
                "sha1": pdf_sha1,
                "markdown_path": str(md_path),
                "chunk_json_path": str(chunk_json_path),
                "faiss_path": faiss_path,
                "chunk_count": chunk_count
            }
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            return {
                "success": False,
                "message": f"处理失败: {str(e)}",
                "file_name": pdf_path.name if pdf_path.exists() else "unknown",
                "error": str(e),
                "traceback": error_traceback
            }
    
    async def process_uploaded_file(
        self,
        file_content: bytes,
        file_name: str,
        company_name: Optional[str] = None,
        chunk_size: int = 30,
        chunk_overlap: int = 5
    ) -> Dict:
        """
        处理上传的 PDF 文件（从内存中的文件内容）
        
        Args:
            file_content: 文件内容（字节）
            file_name: 文件名
            company_name: 公司名称（可选）
            chunk_size: 每个 chunk 的最大行数
            chunk_overlap: chunk 之间的重叠行数
            
        Returns:
            处理结果字典
        """
        # 创建临时文件保存上传的内容
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        try:
            # 将文件移动到 documents 目录
            pdf_path = self.paths.documents_dir / file_name
            shutil.move(tmp_path, pdf_path)
            
            # 处理文件
            result = await self.process_pdf_file(
                pdf_file_path=str(pdf_path),
                company_name=company_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            return result
            
        except Exception as e:
            # 清理临时文件
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()
            
            import traceback
            error_traceback = traceback.format_exc()
            return {
                "success": False,
                "message": f"处理上传文件失败: {str(e)}",
                "file_name": file_name,
                "error": str(e),
                "traceback": error_traceback
            }
