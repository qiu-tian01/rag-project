"""
文档模型
"""
from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class Document(BaseModel):
    """文档模型"""
    document_name: str
    file_path: str
    file_type: str  # docx, pdf, txt, md
    content: str
    pages: Optional[List[Dict[str, Any]]] = None  # 存储每页的内容和页码
    sections: List[str]  # 章节路径列表


class DocumentMetadata(BaseModel):
    """文档元数据"""
    document_name: str
    file_path: str
    file_type: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

