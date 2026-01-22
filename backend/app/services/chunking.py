"""
文档切分服务，基于 LangChain 的 RecursiveCharacterTextSplitter 进行文本切分。
每个 chunk 会携带文档名称、章节路径、页面号以及位置区间，便于后续检索与引用。

参考 RAG-cy/src/text_splitter.py 的实现方式。
"""

from typing import List, Optional, Dict, Any
import uuid
import json
from pathlib import Path
import tiktoken

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models.document import Document
from app.models.chunk import Chunk
from app.utils.hash_utils import calculate_text_sha1


class DocumentChunker:
    """使用 LangChain RecursiveCharacterTextSplitter 的文档切分器"""

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50, model_name: str = "gpt-4o"):
        """
        Args:
            chunk_size: 每段 chunk 大小（token），默认 300（参考 RAG-cy）
            chunk_overlap: chunk 之间重叠的 token 数，默认 50（参考 RAG-cy）
            model_name: 用于 tokenizer 的模型名称，默认 "gpt-4o"（参考 RAG-cy）
        """
        # 使用 RecursiveCharacterTextSplitter.from_tiktoken_encoder 进行智能分割
        # 优先按段落（\n\n）、换行（\n）、空格（ ）分割，尽量保持语义完整性
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: Document) -> List[Chunk]:
        """根据文档内容创建 chunk，优先保留分页信息"""
        chunks: List[Chunk] = []
        section_path = list(document.sections) if document.sections else []

        if document.pages:
            for page in document.pages:
                page_num = page.get("page_num")
                chunks.extend(self._chunk_text(page["text"], document.document_name, section_path, page_num))
        else:
            chunks.extend(self._chunk_text(document.content, document.document_name, section_path))

        return chunks

    def _chunk_text(
        self,
        text: str,
        document_name: str,
        section_path: List[str],
        page_num: Optional[int] = None
    ) -> List[Chunk]:
        """
        使用 RecursiveCharacterTextSplitter 生成 chunk，并补充元数据
        
        参考 RAG-cy 的实现方式，使用智能递归分割策略：
        - 优先按段落（\n\n）、换行（\n）、空格（ ）分割
        - 使用 tiktoken 精确计算 token 数
        - 支持块重叠（chunk_overlap）保持上下文连贯性
        """
        if not text:
            return []

        split_texts = self.splitter.split_text(text)
        if not split_texts:
            return []

        result: List[Chunk] = []
        prev_end = 0
        for chunk_text in split_texts:
            if not chunk_text.strip():
                continue

            start = self._find_chunk_start(text, chunk_text, prev_end)
            end = start + len(chunk_text)
            prev_end = end

            # 计算 token 数量
            length_tokens = self.count_tokens(chunk_text)

            result.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    document_name=document_name,
                    text=chunk_text,
                    section_path=section_path,
                    position={"start": start, "end": end},
                    page_num=page_num,
                    metadata={
                        "model_name": self.model_name,
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                        "length_tokens": length_tokens
                    }
                )
            )

        return result

    @staticmethod
    def _find_chunk_start(text: str, chunk_text: str, prev_end: int) -> int:
        """尝试从上一个 chunk 位置上下浮动寻找 chunk 的真实起点"""
        search_start = max(0, prev_end - len(chunk_text))
        start = text.find(chunk_text, search_start)
        if start == -1:
            start = prev_end
        return start
    
    def chunk_markdown_file(self, md_path: str, chunk_size: int = 30, chunk_overlap: int = 5) -> List[Dict[str, Any]]:
        """
        从 Markdown 文件读取并切分，返回符合 RAG-cy 格式的 chunks
        
        Args:
            md_path: Markdown 文件路径
            chunk_size: 每个分块的最大行数（默认 30）
            chunk_overlap: 分块重叠行数（默认 5）
            
        Returns:
            chunks 列表，每个 chunk 包含 lines 和 text 字段
        """
        md_path = Path(md_path)
        if not md_path.exists():
            raise FileNotFoundError(f"Markdown 文件不存在: {md_path}")
        
        with open(md_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        chunks = []
        i = 0
        total_lines = len(lines)
        
        while i < total_lines:
            start_line = i + 1  # 行号从 1 开始
            end_line = min(i + chunk_size, total_lines)
            chunk_text = ''.join(lines[i:end_line])
            
            if chunk_text.strip():  # 跳过空 chunk
                chunks.append({
                    'lines': [start_line, end_line],
                    'text': chunk_text
                })
            
            i += chunk_size - chunk_overlap
        
        return chunks
    
    def count_tokens(self, text: str, encoding_name: Optional[str] = None) -> int:
        """
        统计文本的 token 数量
        
        参考 RAG-cy 的实现，使用 tiktoken 精确计算 token 数。
        根据 model_name 自动选择对应的编码：
        - gpt-4o 使用 "o200k_base"
        - gpt-3.5/gpt-4 使用 "cl100k_base"
        
        Args:
            text: 文本内容
            encoding_name: tokenizer 编码名称，如果为 None 则根据 model_name 自动选择
            
        Returns:
            token 数量
        """
        if encoding_name is None:
            # 根据 model_name 选择编码
            if "gpt-4o" in self.model_name.lower():
                encoding_name = "o200k_base"
            else:
                encoding_name = "cl100k_base"
        
        try:
            encoding = tiktoken.get_encoding(encoding_name)
            return len(encoding.encode(text))
        except Exception:
            # 如果编码不存在，使用简单估算（1 token ≈ 4 字符）
            return len(text) // 4
    
    def save_chunks_to_json(
        self,
        chunks: List[Dict[str, Any]],
        output_path: str,
        file_name: str,
        sha1: Optional[str] = None,
        company_name: Optional[str] = None
    ) -> str:
        """
        将 chunks 保存为符合 RAG-cy 格式的 JSON 文件
        
        Args:
            chunks: chunks 列表，每个包含 lines 和 text 字段
            output_path: 输出 JSON 文件路径
            file_name: 原始文件名
            sha1: SHA1 哈希值，如果为 None 则基于 chunks 内容计算
            company_name: 公司名称（可选）
            
        Returns:
            保存的文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 如果没有提供 SHA1，基于所有 chunks 的文本内容计算
        if sha1 is None:
            all_text = ''.join(chunk['text'] for chunk in chunks)
            sha1 = calculate_text_sha1(all_text)
        
        # 构建 metainfo
        metainfo = {
            "sha1": sha1,
            "file_name": file_name
        }
        if company_name:
            metainfo["company_name"] = company_name
        
        # 构建符合 RAG-cy 格式的 JSON
        result = {
            "metainfo": metainfo,
            "content": {
                "chunks": chunks
            }
        }
        
        # 保存 JSON 文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return str(output_path)
    
    def chunk_markdown_and_save(
        self,
        md_path: str,
        output_path: str,
        chunk_size: int = 30,
        chunk_overlap: int = 5,
        sha1: Optional[str] = None,
        company_name: Optional[str] = None
    ) -> str:
        """
        从 Markdown 文件切分并保存为 JSON（便捷方法）
        
        Args:
            md_path: Markdown 文件路径
            output_path: 输出 JSON 文件路径
            chunk_size: 每个分块的最大行数
            chunk_overlap: 分块重叠行数
            sha1: SHA1 哈希值（可选）
            company_name: 公司名称（可选）
            
        Returns:
            保存的文件路径
        """
        chunks = self.chunk_markdown_file(md_path, chunk_size, chunk_overlap)
        file_name = Path(md_path).name
        
        return self.save_chunks_to_json(
            chunks=chunks,
            output_path=output_path,
            file_name=file_name,
            sha1=sha1,
            company_name=company_name
        )