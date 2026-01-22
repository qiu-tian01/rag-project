"""
元数据存储（JSON）
"""
import json
import os
import logging
from typing import List, Dict, Optional, Set
from pathlib import Path
from app.models.chunk import Chunk

logger = logging.getLogger(__name__)

class MetadataStorage:
    """元数据存储类，用于持久化存储 chunk 的文本和相关信息"""
    
    def __init__(self, metadata_path: str = None, chunked_reports_dir: str = None):
        """
        初始化元数据存储
        
        Args:
            metadata_path: 元数据文件路径
            chunked_reports_dir: chunk JSON文件目录（用于获取文档SHA1）
        """
        self.metadata_path = metadata_path or os.getenv(
            "METADATA_PATH",
            "./data/metadata/chunks.json"
        )
        self.chunked_reports_dir = chunked_reports_dir or os.getenv(
            "CHUNKED_REPORTS_DIR",
            "./data/metadata/chunked_reports"
        )
        self.chunks: Dict[str, Dict] = {}
        self.load_from_file()
    
    def save_chunk(self, chunk: Chunk):
        """
        保存 chunk 元数据
        
        Args:
            chunk: chunk 对象
        """
        self.chunks[chunk.chunk_id] = chunk.model_dump()
        self.save_to_file()
    
    def get_chunk(self, chunk_id: str) -> Optional[Dict]:
        """
        获取 chunk 元数据
        
        如果chunks字典中没有，尝试从chunk JSON文件中动态加载
        
        Args:
            chunk_id: chunk ID (格式: {sha1}_{index})
            
        Returns:
            chunk 字典
        """
        # 先从内存中查找
        if chunk_id in self.chunks:
            return self.chunks.get(chunk_id)
        
        # 如果内存中没有，尝试从chunk JSON文件中加载
        # chunk_id格式: {sha1}_{index}
        if '_' in chunk_id:
            try:
                sha1, index_str = chunk_id.rsplit('_', 1)
                index = int(index_str)
                
                # 在chunked_reports目录中查找对应的JSON文件
                chunked_reports_path = Path(self.chunked_reports_dir)
                if chunked_reports_path.exists():
                    for json_file in chunked_reports_path.glob("*.json"):
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            
                            file_sha1 = data.get("metainfo", {}).get("sha1")
                            if file_sha1 == sha1:
                                # 找到匹配的文件，提取chunk
                                chunks_data = data.get("content", {}).get("chunks", [])
                                if 0 <= index < len(chunks_data):
                                    chunk_data = chunks_data[index]
                                    text = chunk_data.get("text", "")
                                    
                                    if text and text.strip():
                                        file_name = data.get("metainfo", {}).get("file_name", json_file.stem)
                                        document_name = Path(file_name).stem if file_name else json_file.stem
                                        
                                        # 提取页码
                                        page_num = None
                                        if "page" in chunk_data:
                                            page_num = chunk_data["page"]
                                        else:
                                            import re
                                            page_match = re.search(r'#\s*第\s*(\d+)\s*页', text)
                                            if page_match:
                                                page_num = int(page_match.group(1))
                                        
                                        # 提取lines作为position
                                        lines = chunk_data.get("lines", [])
                                        position = {
                                            "start": lines[0] if len(lines) > 0 else 0,
                                            "end": lines[1] if len(lines) > 1 else lines[0] if len(lines) > 0 else 0
                                        }
                                        
                                        # 构建chunk字典
                                        chunk_dict = {
                                            "chunk_id": chunk_id,
                                            "document_name": document_name,
                                            "section_path": [],
                                            "text": text,
                                            "position": position,
                                            "page_num": page_num,
                                            "metadata": {
                                                "sha1": sha1,
                                                "file_name": file_name,
                                                "lines": lines
                                            }
                                        }
                                        
                                        # 保存到内存中，避免下次再次查找
                                        self.chunks[chunk_id] = chunk_dict
                                        logger.debug(f"[MetadataStorage] 从JSON文件动态加载chunk: {chunk_id}")
                                        return chunk_dict
                        except Exception as e:
                            logger.debug(f"[MetadataStorage] 读取JSON文件失败 {json_file.name}: {e}")
                            continue
            except (ValueError, AttributeError) as e:
                logger.debug(f"[MetadataStorage] 解析chunk_id失败: {chunk_id}, 错误: {e}")
        
        return None
    
    def get_chunks(self, chunk_ids: List[str]) -> List[Dict]:
        """
        批量获取 chunk 元数据
        
        Args:
            chunk_ids: chunk ID 列表
            
        Returns:
            chunk 列表
        """
        return [self.chunks[cid] for cid in chunk_ids if cid in self.chunks]
    
    def get_all_document_names(self) -> Set[str]:
        """
        获取所有文档名称列表
        
        Returns:
            文档名称集合
        """
        document_names = set()
        for chunk in self.chunks.values():
            doc_name = chunk.get('document_name')
            if doc_name:
                document_names.add(doc_name)
        return document_names
    
    def get_chunks_by_document_name(self, document_name: str, fuzzy_match: bool = False) -> List[Dict]:
        """
        根据文档名称获取chunks
        
        Args:
            document_name: 文档名称（支持模糊匹配）
            fuzzy_match: 是否使用模糊匹配（如果为True，只要文档名称包含该字符串即可）
            
        Returns:
            匹配的chunk列表
        """
        matched_chunks = []
        for chunk in self.chunks.values():
            chunk_doc_name = chunk.get('document_name', '')
            if fuzzy_match:
                if document_name in chunk_doc_name or chunk_doc_name in document_name:
                    matched_chunks.append(chunk)
            else:
                if chunk_doc_name == document_name:
                    matched_chunks.append(chunk)
        return matched_chunks
    
    def get_chunk_ids_by_document_name(self, document_name: str, fuzzy_match: bool = False) -> List[str]:
        """
        根据文档名称获取chunk IDs
        
        Args:
            document_name: 文档名称（支持模糊匹配）
            fuzzy_match: 是否使用模糊匹配
            
        Returns:
            匹配的chunk ID列表
        """
        matched_chunks = self.get_chunks_by_document_name(document_name, fuzzy_match)
        return [chunk['chunk_id'] for chunk in matched_chunks]
    
    def get_document_sha1_by_name(self, document_name: str, fuzzy_match: bool = False) -> Optional[str]:
        """
        根据文档名称获取SHA1（用于FAISS索引）
        
        从chunked_reports目录下的JSON文件中查找对应的SHA1
        
        Args:
            document_name: 文档名称
            fuzzy_match: 是否使用模糊匹配
            
        Returns:
            SHA1字符串，如果未找到则返回None
        """
        chunked_reports_path = Path(self.chunked_reports_dir)
        if not chunked_reports_path.exists():
            return None
        
        # 遍历所有JSON文件
        for json_file in chunked_reports_path.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                
                metainfo = report_data.get("metainfo", {})
                file_name = metainfo.get("file_name", "")
                
                # 检查是否匹配
                if fuzzy_match:
                    if document_name in file_name or file_name in document_name:
                        return metainfo.get("sha1")
                else:
                    # 去掉扩展名比较
                    file_name_no_ext = Path(file_name).stem
                    doc_name_no_ext = Path(document_name).stem
                    if file_name_no_ext == doc_name_no_ext or file_name == document_name:
                        return metainfo.get("sha1")
            except Exception:
                continue
        
        return None
    
    def save_to_file(self):
        """保存元数据到 JSON 文件"""
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
    
    def load_from_file(self):
        """从 JSON 文件加载元数据"""
        logger.info(f"[MetadataStorage] 尝试从文件加载chunks: {self.metadata_path}")
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.chunks = json.load(f)
                logger.info(f"[MetadataStorage] 成功加载 {len(self.chunks)} 个chunks")
                if len(self.chunks) == 0:
                    logger.warning(f"[MetadataStorage] chunks.json文件存在但为空！")
                # 统计文档名称
                doc_names = self.get_all_document_names()
                logger.info(f"[MetadataStorage] 包含文档: {list(doc_names)}")
            except Exception as e:
                logger.error(f"[MetadataStorage] 加载chunks失败: {e}", exc_info=True)
                self.chunks = {}
        else:
            logger.warning(f"[MetadataStorage] chunks.json文件不存在: {self.metadata_path}")
            logger.warning(f"[MetadataStorage] 如果chunk JSON文件在chunked_reports目录，需要先加载到MetadataStorage")
            self.chunks = {}

