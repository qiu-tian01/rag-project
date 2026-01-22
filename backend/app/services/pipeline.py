"""
总控 Pipeline 服务
串联解析、分块、向量化、检索和生成流程
"""
import logging
import time
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from pathlib import Path
import os
from tqdm import tqdm

from app.services.retrieval import RetrievalService
from app.services.llm import LLMService
from app.services.embedding import EmbeddingService
from app.services.pdf_to_markdown import PDFToMarkdownService
from app.services.chunking import DocumentChunker
from app.services.vector_db import VectorDBService
from app.utils.parser import DocumentParser
from app.storage.metadata import MetadataStorage
from app.utils.hash_utils import calculate_file_sha1
import json

logger = logging.getLogger(__name__)

class StructuredAnswer(BaseModel):
    """结构化回答模型"""
    answer: str = Field(description="问题的最终答案")
    thoughts: str = Field(description="回答问题的推理过程或摘要")
    citations: List[int] = Field(description="引用内容的页码列表")

class PipelinePaths:
    """Pipeline 路径配置类"""
    
    def __init__(
        self,
        base_dir: str = "./data",
        documents_dir: str = "documents",
        markdown_dir: str = "debug_data",
        chunked_reports_dir: str = "metadata/chunked_reports",
        vector_dbs_dir: str = "metadata/vector_dbs"
    ):
        """
        初始化路径配置
        
        Args:
            base_dir: 基础数据目录
            documents_dir: PDF 文档目录（相对于 base_dir）
            markdown_dir: Markdown 文件目录（相对于 base_dir）
            chunked_reports_dir: Chunk JSON 文件目录（相对于 base_dir）
            vector_dbs_dir: FAISS 索引文件目录（相对于 base_dir）
        """
        self.base_dir = Path(base_dir)
        self.documents_dir = self.base_dir / documents_dir
        self.markdown_dir = self.base_dir / markdown_dir
        self.chunked_reports_dir = self.base_dir / chunked_reports_dir
        self.vector_dbs_dir = self.base_dir / vector_dbs_dir
        
        # 确保所有目录存在
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        self.chunked_reports_dir.mkdir(parents=True, exist_ok=True)
        self.vector_dbs_dir.mkdir(parents=True, exist_ok=True)


class RAGPipeline:
    """RAG 流程总控类"""
    
    def __init__(self, paths: Optional[PipelinePaths] = None):
        """
        初始化 RAG Pipeline
        
        Args:
            paths: 路径配置对象，如果为 None 则使用默认路径
        """
        self.paths = paths or PipelinePaths()
        self.metadata_storage = MetadataStorage(
            chunked_reports_dir=str(self.paths.chunked_reports_dir)
        )
        self.retrieval_service = RetrievalService()
        self.llm_service = LLMService()
        self.embedding_service = EmbeddingService()
        self.parser = DocumentParser()
        self.chunker = DocumentChunker()
        self.pdf_to_markdown = PDFToMarkdownService()
        self.vector_db = VectorDBService(self.embedding_service)
    
    async def answer(
        self, 
        query: str, 
        history: List[Dict[str, str]] = None,
        search_mode: int = 2,
        llm_model: int = 2,
        product_name: Optional[str] = None
    ) -> Dict:
        """
        完整 RAG 问答流程，包含结构化输出和引用验证 (异步)
        
        Args:
            query: 用户查询
            history: 对话历史
            search_mode: 搜索模式，1=纯向量搜索，2=混合检索+rerank
            llm_model: 大模型选择，1=qwen-max, 2=qwen-plus, 3=qwen-turbo
            product_name: 可选的产品名称，用于过滤相关文档
        """
        pipeline_start = time.time()
        logger.info("[Pipeline] 开始处理 RAG 问答流程")
        
        # 0. 设置LLM模型
        step_start = time.time()
        self.llm_service.set_model(llm_model)
        model_name = {1: "qwen-max", 2: "qwen-plus", 3: "qwen-turbo"}.get(llm_model, "qwen-plus")
        logger.info(f"[Pipeline] 步骤0: 设置LLM模型 -> {model_name} (耗时: {time.time() - step_start:.2f}秒)")
        
        # 1. 产品名称提取（如果未提供）
        step_start = time.time()
        final_product_name = product_name
        if not final_product_name:
            # 尝试从查询中提取产品名称（简单关键词匹配）
            # 获取所有文档名称
            all_doc_names = self.metadata_storage.get_all_document_names()
            logger.debug(f"[Pipeline] 可用文档名称: {list(all_doc_names)[:5]}...")
            for doc_name in all_doc_names:
                # 如果查询中包含文档名称的关键词，使用该文档
                if doc_name in query or any(keyword in query for keyword in doc_name.split() if len(keyword) > 2):
                    final_product_name = doc_name
                    logger.info(f"[Pipeline] 从查询中提取到产品名称: {final_product_name}")
                    break
        else:
            logger.info(f"[Pipeline] 使用提供的产品名称: {final_product_name}")
        logger.info(f"[Pipeline] 步骤1: 产品名称提取 -> {final_product_name or '未指定'} (耗时: {time.time() - step_start:.2f}秒)")
        
        # 2. 获取文档SHA1（如果提供了产品名称）
        step_start = time.time()
        document_sha1 = None
        if final_product_name:
            document_sha1 = self.metadata_storage.get_document_sha1_by_name(final_product_name, fuzzy_match=True)
            if document_sha1:
                logger.info(f"[Pipeline] 找到文档SHA1: {document_sha1[:16]}...")
            else:
                logger.warning(f"[Pipeline] 未找到产品名称 '{final_product_name}' 对应的文档SHA1")
        logger.info(f"[Pipeline] 步骤2: 获取文档SHA1 -> {document_sha1[:16] + '...' if document_sha1 else '无'} (耗时: {time.time() - step_start:.2f}秒)")
        
        # 3. 查询改写 (可选)
        step_start = time.time()
        optimized_query = await self.llm_service.rewrite_query(query)
        logger.info(f"[Pipeline] 步骤3: 查询改写")
        logger.debug(f"[Pipeline] 原始查询: {query[:100]}...")
        logger.debug(f"[Pipeline] 优化查询: {optimized_query[:100]}...")
        logger.info(f"[Pipeline] 步骤3耗时: {time.time() - step_start:.2f}秒")
        
        # 4. 检索相关上下文
        step_start = time.time()
        logger.info(f"[Pipeline] 步骤4: 开始检索 (模式={search_mode}, top_k=10)")
        search_results = await self.retrieval_service.search(
            optimized_query,
            top_k=10,
            search_mode=search_mode,
            product_name=final_product_name,
            document_sha1=document_sha1
        )
        logger.info(f"[Pipeline] 步骤4: 检索完成，获得 {len(search_results)} 个结果 (耗时: {time.time() - step_start:.2f}秒)")
        if search_results:
            logger.debug(f"[Pipeline] 检索结果示例: {search_results[0].get('document_name', 'N/A')} (相似度: {search_results[0].get('similarity', 0):.3f})")
        
        # 5. 组装 Prompt，包含结构化要求
        step_start = time.time()
        context_items = []
        available_pages = set()
        for i, res in enumerate(search_results):
            page_info = f", 第 {res['page_num']} 页" if res.get('page_num') else ""
            section_info = " > ".join(res.get('section_path', [])) if res.get('section_path') else ""
            section_str = f" [{section_info}]" if section_info else ""
            context_items.append(f"[{i+1}] 来自《{res['document_name']}》{section_str}{page_info}:\n{res['text']}")
            if res.get('page_num'):
                available_pages.add(res['page_num'])
                
        context = "\n\n".join(context_items)
        logger.info(f"[Pipeline] 步骤5: 组装上下文 (总长度: {len(context)} 字符, 可用页码: {sorted(available_pages)}) (耗时: {time.time() - step_start:.2f}秒)")
        
        system_prompt = """你是一个专业的需求分析助手。请基于提供的参考内容回答用户问题。
要求：
1. 必须严格基于参考内容回答，不要编造。
2. 给出推理过程（thoughts）。
3. 列出引用的页码（citations）。
4. 输出必须是 JSON 格式，包含字段：answer, thoughts, citations。
"""
        
        prompt = f"参考内容：\n{context}\n\n用户问题：{query}\n\n请以 JSON 格式输出回答。"
        
        # 6. 生成答案（使用选定的模型）
        step_start = time.time()
        logger.info(f"[Pipeline] 步骤6: 调用LLM生成答案 (模型: {model_name})")
        raw_answer = await self.llm_service.generate(prompt, system_prompt=system_prompt)
        logger.info(f"[Pipeline] 步骤6: LLM生成完成 (答案长度: {len(raw_answer)} 字符, 耗时: {time.time() - step_start:.2f}秒)")
        
        try:
            # 尝试解析 JSON
            step_start = time.time()
            clean_json = raw_answer.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:-3].strip()
            elif clean_json.startswith("```"):
                clean_json = clean_json[3:-3].strip()
                
            answer_dict = json.loads(clean_json)
            logger.info(f"[Pipeline] 步骤7: JSON解析成功 (耗时: {time.time() - step_start:.2f}秒)")
            
            # 7. 引用验证
            step_start = time.time()
            original_citations = answer_dict.get('citations', [])
            valid_citations = [p for p in original_citations if p in available_pages]
            answer_dict['citations'] = valid_citations
            if len(original_citations) != len(valid_citations):
                logger.warning(f"[Pipeline] 引用验证: 原始引用 {len(original_citations)} 个，有效引用 {len(valid_citations)} 个")
            else:
                logger.info(f"[Pipeline] 步骤7: 引用验证完成，有效引用 {len(valid_citations)} 个 (耗时: {time.time() - step_start:.2f}秒)")
            
            total_time = time.time() - pipeline_start
            logger.info(f"[Pipeline] RAG流程全部完成，总耗时: {total_time:.2f}秒")
            logger.info("-" * 80)
            
            # 确保thoughts是字符串类型
            thoughts = answer_dict.get('thoughts')
            if isinstance(thoughts, list):
                # 如果是列表，转换为字符串
                thoughts = ' '.join(str(item) for item in thoughts) if thoughts else None
            elif thoughts is not None:
                thoughts = str(thoughts)
            
            # 确保citations是整数列表
            citations = answer_dict.get('citations', [])
            if citations:
                # 确保所有元素都是整数
                citations = [int(c) if isinstance(c, (int, str)) and str(c).isdigit() else c for c in citations]
                citations = [c for c in citations if isinstance(c, int)]
            
            return {
                "answer": str(answer_dict.get('answer', '')),
                "thoughts": thoughts,
                "citations": citations,
                "sources": search_results
            }
        except Exception as e:
            logger.error(f"[Pipeline] JSON解析失败: {str(e)}")
            logger.debug(f"[Pipeline] 原始答案: {raw_answer[:200]}...")
            total_time = time.time() - pipeline_start
            logger.warning(f"[Pipeline] 使用原始答案返回，总耗时: {total_time:.2f}秒")
            logger.info("-" * 80)
            return {
                "answer": raw_answer,
                "thoughts": "解析结构化输出失败",
                "citations": [],
                "sources": search_results
            }

    async def ingest_directory(self, directory_path: str):
        """
        离线入库流程 (支持异步)
        """
        import os
        
        all_chunks = []
        all_embeddings = []
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith(('.pdf', '.docx', '.txt', '.md')):
                    file_path = os.path.join(root, file)
                    print(f"正在处理文件: {file_path}")
                    
                    try:
                        # 1. 解析
                        doc = self.parser.parse(file_path)
                        # 2. 切分
                        chunks = self.chunker.chunk_document(doc)
                        
                        # 3. 生成 Embeddings (批量异步)
                        texts = [c.text for c in chunks]
                        embeddings = await self.embedding_service.embed_documents(texts)
                        
                        # 4. 暂存
                        for chunk, emb in zip(chunks, embeddings):
                            self.metadata_storage.save_chunk(chunk)
                            all_chunks.append(chunk)
                            all_embeddings.append(emb)
                    except Exception as e:
                        print(f"处理文件 {file_path} 失败: {e}")
        
        # 5. 构建并保存索引
        chunk_ids = [c.chunk_id for c in all_chunks]
        self.retrieval_service.faiss_index.build_index(all_embeddings, chunk_ids)
        print("入库完成！")
    
    async def process_documents(
        self,
        documents_dir: Optional[str] = None,
        skip_existing: bool = True
    ):
        """
        完整的文档处理流程（参考 RAG-cy 的处理方式）
        
        流程：
        1. PDF → Markdown（保存到 debug_data）
        2. Markdown → Chunks（保存到 metadata/chunked_reports）
        3. Chunks → Embeddings → FAISS（保存到 metadata/vector_dbs）
        
        Args:
            documents_dir: PDF 文档目录，如果为 None 则使用 self.paths.documents_dir
            skip_existing: 是否跳过已处理的文件（基于 SHA1 判断）
        """
        documents_dir = Path(documents_dir) if documents_dir else self.paths.documents_dir
        
        if not documents_dir.exists():
            raise FileNotFoundError(f"文档目录不存在: {documents_dir}")
        
        # 获取所有 PDF 文件
        pdf_files = list(documents_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"在 {documents_dir} 中未找到 PDF 文件")
            return
        
        print(f"找到 {len(pdf_files)} 个 PDF 文件，开始处理...")
        
        processed_count = 0
        failed_count = 0
        
        for pdf_file in tqdm(pdf_files, desc="处理 PDF 文档"):
            try:
                # 计算 PDF 的 SHA1
                pdf_sha1 = calculate_file_sha1(pdf_file)
                
                # 检查是否已处理（如果启用跳过）
                if skip_existing:
                    chunk_json_path = self.paths.chunked_reports_dir / f"{pdf_file.stem}.json"
                    faiss_path = self.paths.vector_dbs_dir / f"{pdf_sha1}.faiss"
                    
                    if chunk_json_path.exists() and faiss_path.exists():
                        print(f"跳过已处理的文件: {pdf_file.name}")
                        continue
                
                # 步骤 1: PDF → Markdown
                print(f"\n[1/3] 转换 PDF 为 Markdown: {pdf_file.name}")
                md_path = self.pdf_to_markdown.convert_pdf_to_markdown(
                    str(pdf_file),
                    str(self.paths.markdown_dir)
                )
                
                # 步骤 2: Markdown → Chunks (保存为 JSON)
                print(f"[2/3] 切分 Markdown 为 Chunks: {pdf_file.name}")
                chunk_json_path = self.paths.chunked_reports_dir / f"{pdf_file.stem}.json"
                self.chunker.chunk_markdown_and_save(
                    md_path=str(md_path),
                    output_path=str(chunk_json_path),
                    sha1=pdf_sha1,
                    company_name=None  # 可以后续从配置文件读取
                )
                
                # 步骤 3: Chunks → Embeddings → FAISS
                print(f"[3/3] 生成向量索引: {pdf_file.name}")
                await self.vector_db.process_chunk_json(
                    chunk_json_path=str(chunk_json_path),
                    output_dir=str(self.paths.vector_dbs_dir)
                )
                
                processed_count += 1
                print(f"✓ 完成处理: {pdf_file.name}")
                
            except Exception as e:
                failed_count += 1
                print(f"✗ 处理失败 {pdf_file.name}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n处理完成！成功: {processed_count}, 失败: {failed_count}")


async def process_single_pdf(
    pdf_file_path: str,
    base_dir: str = "./data",
    company_name: Optional[str] = None,
    chunk_size: int = 30,
    chunk_overlap: int = 5
):
    """
    处理单个 PDF 文件的完整流程
    
    流程：
    1. PDF → Markdown（保存到 debug_data）
    2. Markdown → Chunks（保存到 metadata/chunked_reports）
    3. Chunks → Embeddings → FAISS（保存到 metadata/vector_dbs）
    
    Args:
        pdf_file_path: PDF 文件路径
        base_dir: 基础数据目录
        company_name: 公司名称（可选）
        chunk_size: 每个 chunk 的最大行数
        chunk_overlap: chunk 之间的重叠行数
    """
    import asyncio
    
    print("=" * 60)
    print("文档处理流程")
    print("=" * 60)
    
    pdf_path = Path(pdf_file_path)
    if not pdf_path.exists():
        print(f"❌ 错误：PDF 文件不存在: {pdf_file_path}")
        return
    
    print(f"📄 PDF 文件: {pdf_path.name}")
    print(f"📁 文件大小: {pdf_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"📂 基础目录: {base_dir}\n")
    
    # 配置路径
    paths = PipelinePaths(
        base_dir=base_dir,
        documents_dir="documents",
        markdown_dir="debug_data",
        chunked_reports_dir="metadata/chunked_reports",
        vector_dbs_dir="metadata/vector_dbs"
    )
    
    # 创建 Pipeline 实例
    pipeline = RAGPipeline(paths=paths)
    
    try:
        # 计算 PDF 的 SHA1
        pdf_sha1 = calculate_file_sha1(pdf_path)
        print(f"🔐 SHA1: {pdf_sha1}\n")
        
        # 步骤 1: PDF → Markdown
        print("-" * 60)
        print("步骤 1/4: PDF → Markdown 转换")
        print("-" * 60)
        md_path = pipeline.pdf_to_markdown.convert_pdf_to_markdown(
            str(pdf_path),
            str(paths.markdown_dir)
        )
        print(f"✅ Markdown 文件已保存: {md_path}")
        
        # 步骤 2: Markdown → Chunks
        print("\n" + "-" * 60)
        print("步骤 2/4: Markdown → Chunks 切分")
        print("-" * 60)
        chunk_json_path = paths.chunked_reports_dir / f"{pdf_path.stem}.json"
        pipeline.chunker.chunk_markdown_and_save(
            md_path=str(md_path),
            output_path=str(chunk_json_path),
            sha1=pdf_sha1,
            company_name=company_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # 读取并显示 chunk 统计信息
        with open(chunk_json_path, 'r', encoding='utf-8') as f:
            chunk_data = json.load(f)
        
        chunk_count = len(chunk_data['content']['chunks'])
        print(f"✅ Chunk JSON 文件已保存: {chunk_json_path}")
        print(f"   - 共生成 {chunk_count} 个 chunks")
        print(f"   - SHA1: {chunk_data['metainfo']['sha1']}")
        
        # 步骤 3: Chunks → Embeddings
        print("\n" + "-" * 60)
        print("步骤 3/4: Chunks → Embeddings 生成")
        print("-" * 60)
        print(f"   正在为 {chunk_count} 个 chunks 生成 embeddings...")
        
        # 提取所有 chunk 文本
        chunks = chunk_data['content']['chunks']
        texts = [chunk['text'] for chunk in chunks]
        
        embeddings = await pipeline.embedding_service.embed_documents(texts)
        print(f"✅ 成功生成 {len(embeddings)} 个 embeddings")
        print(f"   - Embedding 维度: {len(embeddings[0])}")
        
        # 步骤 4: 创建并保存 FAISS 索引
        print("\n" + "-" * 60)
        print("步骤 4/4: 创建并保存 FAISS 向量索引")
        print("-" * 60)
        faiss_path = await pipeline.vector_db.process_chunk_json(
            chunk_json_path=str(chunk_json_path),
            output_dir=str(paths.vector_dbs_dir)
        )
        
        print(f"✅ FAISS 索引文件已保存: {faiss_path}")
        
        # 验证索引文件
        faiss_file = Path(faiss_path)
        if faiss_file.exists():
            file_size = faiss_file.stat().st_size / 1024
            print(f"   - 索引文件大小: {file_size:.2f} KB")
        
        print("\n" + "=" * 60)
        print("✅ 所有处理步骤完成！")
        print("=" * 60)
        print(f"\n生成的文件:")
        print(f"  📝 Markdown: {md_path}")
        print(f"  📦 Chunk JSON: {chunk_json_path}")
        print(f"  🔍 FAISS 索引: {faiss_path}")
        print(f"\n统计信息:")
        print(f"  - Chunk 数量: {chunk_count}")
        print(f"  - Embedding 维度: {len(embeddings[0])}")
        print(f"  - SHA1: {pdf_sha1}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    import asyncio
    from pathlib import Path
    
    # 添加项目根目录到 Python 路径，确保可以导入 app 模块
    # 获取当前文件的目录（app/services/）
    current_file = Path(__file__).resolve()
    # 获取 backend 目录（项目根目录）
    backend_dir = current_file.parent.parent.parent
    # 添加到 sys.path
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    
    # 解析命令行参数
    base_dir = "./data"
    documents_dir = None
    skip_existing = True
    
    # 解析可选参数
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--base-dir" and i + 1 < len(sys.argv):
            base_dir = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--documents-dir" and i + 1 < len(sys.argv):
            documents_dir = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--no-skip-existing":
            skip_existing = False
            i += 1
        elif sys.argv[i] == "--help" or sys.argv[i] == "-h":
            print("使用方法:")
            print("  python -m app.services.pipeline [options]")
            print("\n参数:")
            print("  --base-dir        基础数据目录（默认: ./data）")
            print("  --documents-dir   PDF 文档目录（相对于 base_dir，默认: documents）")
            print("  --no-skip-existing  不跳过已处理的文件（默认会跳过）")
            print("  --help, -h        显示帮助信息")
            print("\n说明:")
            print("  批量处理指定目录下的所有 PDF 文档，执行以下流程：")
            print("  1. PDF → Markdown（保存到 debug_data）")
            print("  2. Markdown → Chunks（保存到 metadata/chunked_reports）")
            print("  3. Chunks → Embeddings → FAISS（保存到 metadata/vector_dbs）")
            print("\n示例:")
            print("  python -m app.services.pipeline")
            print("  python -m app.services.pipeline --base-dir ./custom_data")
            print("  python -m app.services.pipeline --documents-dir custom_docs")
            print("  python -m app.services.pipeline --no-skip-existing")
            sys.exit(0)
        else:
            print(f"⚠️  未知参数: {sys.argv[i]}")
            print("  使用 --help 查看帮助信息")
            i += 1
    
    # 创建 Pipeline 实例
    paths = PipelinePaths(base_dir=base_dir)
    pipeline = RAGPipeline(paths=paths)
    
    # 运行批量处理流程
    print("=" * 60)
    print("批量处理文档流程")
    print("=" * 60)
    print(f"📂 基础目录: {base_dir}")
    print(f"📁 文档目录: {documents_dir or paths.documents_dir}")
    print(f"⏭️  跳过已处理: {skip_existing}")
    print()
    
    try:
        asyncio.run(pipeline.process_documents(
            documents_dir=documents_dir,
            skip_existing=skip_existing
        ))
        print("\n✅ 所有文档处理完成！")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n⚠️  用户中断处理")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

