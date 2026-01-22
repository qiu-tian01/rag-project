"""
PDF 转 Markdown 服务
将 PDF 文件转换为 Markdown 格式，保留页面结构
"""
import os
from pathlib import Path
from typing import Optional
import pdfplumber


class PDFToMarkdownService:
    """PDF 转 Markdown 服务类"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        初始化服务
        
        Args:
            output_dir: Markdown 文件输出目录，默认为 None（需要调用时指定）
        """
        self.output_dir = output_dir
    
    def convert_pdf_to_markdown(
        self, 
        pdf_path: str, 
        output_dir: Optional[str] = None,
        use_pymupdf: bool = False
    ) -> str:
        """
        将 PDF 文件转换为 Markdown 格式
        
        Args:
            pdf_path: PDF 文件路径
            output_dir: 输出目录，如果为 None 则使用 self.output_dir
            use_pymupdf: 是否使用 PyMuPDF（fitz），False 则使用 pdfplumber
            
        Returns:
            Markdown 文件路径
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")
        
        output_dir = Path(output_dir) if output_dir else Path(self.output_dir) if self.output_dir else None
        if not output_dir:
            raise ValueError("必须指定 output_dir")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成输出文件名（保持原文件名，扩展名改为 .md）
        md_filename = pdf_path.stem + ".md"
        md_path = output_dir / md_filename
        
        if use_pymupdf:
            markdown_content = self._convert_with_pymupdf(pdf_path)
        else:
            markdown_content = self._convert_with_pdfplumber(pdf_path)
        
        # 保存 Markdown 文件
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        return str(md_path)
    
    def _convert_with_pdfplumber(self, pdf_path: Path) -> str:
        """
        使用 pdfplumber 转换 PDF 为 Markdown
        
        Args:
            pdf_path: PDF 文件路径
            
        Returns:
            Markdown 内容
        """
        markdown_parts = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    # 添加页面分隔符
                    markdown_parts.append(f"# 第 {i} 页\n\n")
                    markdown_parts.append(text)
                    markdown_parts.append("\n\n")
        
        return "".join(markdown_parts)
    
    def _convert_with_pymupdf(self, pdf_path: Path) -> str:
        """
        使用 PyMuPDF (fitz) 转换 PDF 为 Markdown
        
        Args:
            pdf_path: PDF 文件路径
            
        Returns:
            Markdown 内容
        """
        import fitz  # PyMuPDF
        
        markdown_parts = []
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text:
                # 添加页面分隔符
                markdown_parts.append(f"# 第 {page_num + 1} 页\n\n")
                markdown_parts.append(text)
                markdown_parts.append("\n\n")
        
        doc.close()
        return "".join(markdown_parts)
    
    def convert_directory(
        self, 
        pdf_dir: str, 
        output_dir: str,
        use_pymupdf: bool = False
    ) -> list[str]:
        """
        批量转换目录下的所有 PDF 文件
        
        Args:
            pdf_dir: PDF 文件目录
            output_dir: 输出目录
            use_pymupdf: 是否使用 PyMuPDF
            
        Returns:
            Markdown 文件路径列表
        """
        pdf_dir = Path(pdf_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        md_files = []
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        for pdf_file in pdf_files:
            try:
                md_path = self.convert_pdf_to_markdown(
                    str(pdf_file), 
                    str(output_dir),
                    use_pymupdf=use_pymupdf
                )
                md_files.append(md_path)
                print(f"已转换: {pdf_file.name} -> {Path(md_path).name}")
            except Exception as e:
                print(f"转换失败 {pdf_file.name}: {e}")
        
        return md_files
