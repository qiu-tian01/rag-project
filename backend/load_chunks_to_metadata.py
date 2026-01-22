"""
将chunk JSON文件中的chunks加载到MetadataStorage
"""
import json
import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from app.storage.metadata import MetadataStorage
from app.models.chunk import Chunk

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_chunks_from_json(chunk_json_path: Path, metadata_storage: MetadataStorage):
    """
    从chunk JSON文件加载chunks到MetadataStorage
    
    Args:
        chunk_json_path: chunk JSON文件路径
        metadata_storage: MetadataStorage实例
    """
    try:
        with open(chunk_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metainfo = data.get("metainfo", {})
        sha1 = metainfo.get("sha1")
        file_name = metainfo.get("file_name", chunk_json_path.stem)
        document_name = Path(file_name).stem if file_name else chunk_json_path.stem
        
        chunks_data = data.get("content", {}).get("chunks", [])
        
        logger.info(f"处理文件: {chunk_json_path.name}")
        logger.info(f"  文档名称: {document_name}")
        logger.info(f"  SHA1: {sha1}")
        logger.info(f"  Chunks数量: {len(chunks_data)}")
        
        loaded_count = 0
        skipped_count = 0
        
        for idx, chunk_data in enumerate(chunks_data):
            try:
                # 从chunk数据中提取信息
                text = chunk_data.get("text", "")
                if not text or not text.strip():
                    skipped_count += 1
                    continue
                
                # 生成chunk_id（与vector_db.py中的逻辑保持一致）
                chunk_id = chunk_data.get("chunk_id")
                if not chunk_id:
                    # 使用格式: {sha1}_{index} 作为chunk_id
                    chunk_id = f"{sha1}_{idx}"
                
                # 检查是否已存在
                if chunk_id in metadata_storage.chunks:
                    logger.debug(f"  Chunk {chunk_id} 已存在，跳过")
                    skipped_count += 1
                    continue
                
                # 提取页码（从text中提取，或者从chunk数据中获取）
                page_num = None
                if "page" in chunk_data:
                    page_num = chunk_data["page"]
                else:
                    # 尝试从text中提取页码（例如 "# 第 1 页"）
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
                
                # 创建Chunk对象
                chunk = Chunk(
                    chunk_id=chunk_id,
                    document_name=document_name,
                    section_path=[],  # chunk JSON中没有section_path信息
                    text=text,
                    position=position,
                    page_num=page_num,
                    metadata={
                        "sha1": sha1,
                        "file_name": file_name,
                        "lines": lines
                    }
                )
                
                # 保存到MetadataStorage
                metadata_storage.save_chunk(chunk)
                loaded_count += 1
                
            except Exception as e:
                logger.error(f"  处理chunk {idx} 失败: {e}")
                skipped_count += 1
                continue
        
        logger.info(f"  加载: {loaded_count} 个chunks, 跳过: {skipped_count} 个chunks")
        return loaded_count
        
    except Exception as e:
        logger.error(f"处理文件 {chunk_json_path.name} 失败: {e}", exc_info=True)
        return 0

def main():
    """主函数：加载所有chunk JSON文件到MetadataStorage"""
    chunked_reports_dir = Path("./data/metadata/chunked_reports")
    
    if not chunked_reports_dir.exists():
        logger.error(f"chunked_reports目录不存在: {chunked_reports_dir}")
        return
    
    # 初始化MetadataStorage
    metadata_storage = MetadataStorage()
    logger.info(f"MetadataStorage初始化完成，当前有 {len(metadata_storage.chunks)} 个chunks")
    
    # 获取所有JSON文件
    json_files = list(chunked_reports_dir.glob("*.json"))
    logger.info(f"找到 {len(json_files)} 个chunk JSON文件")
    
    total_loaded = 0
    total_skipped = 0
    
    for json_file in json_files:
        loaded = load_chunks_from_json(json_file, metadata_storage)
        total_loaded += loaded
    
    logger.info("=" * 80)
    logger.info(f"完成！")
    logger.info(f"  总共加载: {total_loaded} 个chunks")
    logger.info(f"  MetadataStorage中现在有: {len(metadata_storage.chunks)} 个chunks")
    
    # 保存到文件
    metadata_storage.save_to_file()
    logger.info(f"  已保存到: {metadata_storage.metadata_path}")

if __name__ == "__main__":
    main()
