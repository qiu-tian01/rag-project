"""
Chunk 模型
"""
from pydantic import BaseModel
from typing import List, Dict, Optional


class Chunk(BaseModel):
    """Chunk 模型"""
    chunk_id: str
    document_name: str
    section_path: List[str]  # 章节路径，如 ["1 总则", "1.2 范围"]
    text: str
    position: Dict[str, int]  # {"start": 0, "end": 100}
    page_num: Optional[int] = None # 增加页码字段
    metadata: Optional[Dict] = None


class ChunkMetadata(BaseModel):
    """Chunk 元数据（用于存储）"""
    chunk_id: str
    document_name: str
    section_path: List[str]
    text: str
    position: Dict[str, int]
    page_num: Optional[int] = None # 增加页码字段
    metadata: Optional[Dict] = None

