"""
SHA1 哈希计算工具
用于生成文档的唯一标识符
"""
import hashlib
from pathlib import Path
from typing import Union


def calculate_file_sha1(file_path: Union[str, Path]) -> str:
    """
    基于文件内容计算 SHA1 哈希值
    
    Args:
        file_path: 文件路径
        
    Returns:
        SHA1 哈希值（十六进制字符串）
    """
    sha1_hash = hashlib.sha1()
    
    with open(file_path, 'rb') as f:
        # 分块读取，避免大文件占用过多内存
        for chunk in iter(lambda: f.read(4096), b""):
            sha1_hash.update(chunk)
    
    return sha1_hash.hexdigest()


def calculate_text_sha1(text: str) -> str:
    """
    基于文本内容计算 SHA1 哈希值
    
    Args:
        text: 文本内容
        
    Returns:
        SHA1 哈希值（十六进制字符串）
    """
    sha1_hash = hashlib.sha1()
    sha1_hash.update(text.encode('utf-8'))
    return sha1_hash.hexdigest()
