"""
批量生成FAISS索引文件
为chunked_reports目录下的所有JSON文件生成对应的FAISS索引
"""
import asyncio
import sys
import json
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from app.services.vector_db import VectorDBService
from app.services.embedding import EmbeddingService

async def generate_all_faiss_indexes():
    """为所有chunk JSON文件生成FAISS索引"""
    chunked_reports_dir = Path("./data/metadata/chunked_reports")
    vector_dbs_dir = Path("./data/metadata/vector_dbs")
    
    if not chunked_reports_dir.exists():
        print(f"错误：chunked_reports目录不存在: {chunked_reports_dir}")
        return
    
    vector_dbs_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有JSON文件
    json_files = list(chunked_reports_dir.glob("*.json"))
    print(f"找到 {len(json_files)} 个chunk JSON文件")
    
    # 初始化服务
    try:
        embedding_service = EmbeddingService()
        vector_db = VectorDBService(embedding_service)
    except Exception as e:
        print(f"初始化服务失败: {e}")
        print("请检查环境变量 QWEN_API_KEY 或 DASHSCOPE_API_KEY 是否设置")
        return
    
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    for json_file in json_files:
        # 读取JSON文件获取SHA1
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                sha1 = data.get("metainfo", {}).get("sha1")
                file_name = data.get("metainfo", {}).get("file_name", json_file.name)
        except Exception as e:
            print(f"读取 {json_file.name} 失败: {e}")
            failed_count += 1
            continue
        
        if not sha1:
            print(f"跳过 {json_file.name}：缺少SHA1")
            skipped_count += 1
            continue
        
        # 检查是否已存在
        faiss_file = vector_dbs_dir / f"{sha1}.faiss"
        if faiss_file.exists():
            print(f"跳过 {json_file.name}：FAISS索引已存在 ({faiss_file.name})")
            skipped_count += 1
            continue
        
        print(f"\n处理: {json_file.name}")
        print(f"  SHA1: {sha1}")
        print(f"  文件名: {file_name}")
        
        try:
            faiss_path = await vector_db.process_chunk_json(
                chunk_json_path=str(json_file),
                output_dir=str(vector_dbs_dir),
                max_chunk_length=2048
            )
            print(f"  [OK] 成功生成: {Path(faiss_path).name}")
            processed_count += 1
        except Exception as e:
            print(f"  [ERROR] 处理失败: {e}")
            failed_count += 1
            import traceback
            traceback.print_exc()
    
    print(f"\n完成！")
    print(f"  处理: {processed_count} 个文件")
    print(f"  跳过: {skipped_count} 个文件")
    print(f"  失败: {failed_count} 个文件")
    print(f"  总计: {len(json_files)} 个文件")

if __name__ == "__main__":
    asyncio.run(generate_all_faiss_indexes())
