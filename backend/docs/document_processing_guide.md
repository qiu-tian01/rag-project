# 文档处理流程使用指南

本文档说明如何使用新的文档处理流程，参考 RAG-cy 项目的处理方式。

## 功能概述

完整的文档处理流程包括以下步骤：

1. **PDF → Markdown**：将 PDF 文件转换为 Markdown 格式
2. **Markdown → Chunks**：将 Markdown 文件切分为 chunks，保存为 JSON
3. **Chunks → Embeddings → FAISS**：生成向量并创建 FAISS 索引

## 文件结构

处理后的文件将按照以下结构存储：

```
backend/data/
├── documents/              # 原始 PDF 文件
├── debug_data/             # Markdown 文件（PDF 转换后）
└── metadata/
    ├── chunked_reports/    # Chunk JSON 文件
    └── vector_dbs/         # FAISS 索引文件（使用 SHA1 命名）
```

## 使用方法

### 基本使用

```python
import asyncio
from app.services.pipeline import RAGPipeline, PipelinePaths

async def main():
    # 创建路径配置（使用默认路径）
    paths = PipelinePaths(
        base_dir="./data",
        documents_dir="documents",
        markdown_dir="debug_data",
        chunked_reports_dir="metadata/chunked_reports",
        vector_dbs_dir="metadata/vector_dbs"
    )
    
    # 创建 Pipeline 实例
    pipeline = RAGPipeline(paths=paths)
    
    # 处理文档
    await pipeline.process_documents(
        documents_dir=None,  # 使用 paths.documents_dir
        skip_existing=True   # 跳过已处理的文件
    )

if __name__ == "__main__":
    asyncio.run(main())
```

### 自定义路径

```python
# 使用自定义路径
paths = PipelinePaths(
    base_dir="/path/to/data",
    documents_dir="pdfs",
    markdown_dir="markdowns",
    chunked_reports_dir="chunks",
    vector_dbs_dir="indices"
)

pipeline = RAGPipeline(paths=paths)
await pipeline.process_documents()
```

### 单独使用各个服务

#### 1. PDF 转 Markdown

```python
from app.services.pdf_to_markdown import PDFToMarkdownService

service = PDFToMarkdownService()
md_path = service.convert_pdf_to_markdown(
    pdf_path="document.pdf",
    output_dir="./output"
)
```

#### 2. Markdown 切分为 Chunks

```python
from app.services.chunking import DocumentChunker
from app.utils.hash_utils import calculate_file_sha1

chunker = DocumentChunker()
sha1 = calculate_file_sha1("document.pdf")

chunk_json_path = chunker.chunk_markdown_and_save(
    md_path="document.md",
    output_path="document.json",
    sha1=sha1,
    company_name="公司名称"  # 可选
)
```

#### 3. 生成向量索引

```python
from app.services.vector_db import VectorDBService
from app.services.embedding import EmbeddingService

embedding_service = EmbeddingService()
vector_db = VectorDBService(embedding_service)

faiss_path = await vector_db.process_chunk_json(
    chunk_json_path="document.json",
    output_dir="./vector_dbs"
)
```

## 数据格式

### Chunk JSON 格式

生成的 chunk JSON 文件格式与 RAG-cy 兼容：

```json
{
  "metainfo": {
    "sha1": "文档的SHA1哈希值",
    "file_name": "文档文件名.md",
    "company_name": "公司名称（可选）"
  },
  "content": {
    "chunks": [
      {
        "lines": [1, 30],
        "text": "chunk 的文本内容"
      }
    ]
  }
}
```

### FAISS 索引文件

- 文件名格式：`{sha1}.faiss`
- 使用内积（IndexFlatIP）进行余弦相似度计算
- 向量已归一化

## 配置说明

### 路径配置

所有路径都可以通过 `PipelinePaths` 类进行配置：

- `base_dir`：基础数据目录
- `documents_dir`：PDF 文档目录
- `markdown_dir`：Markdown 文件目录
- `chunked_reports_dir`：Chunk JSON 文件目录
- `vector_dbs_dir`：FAISS 索引文件目录

### Chunk 参数

在 `DocumentChunker` 中可以配置：

- `chunk_size`：每个 chunk 的最大行数（默认 30）
- `chunk_overlap`：chunk 之间的重叠行数（默认 5）

### Embedding 配置

在 `EmbeddingService` 中配置：

- `model`：使用的 embedding 模型（默认 "text-embedding-v3"）
- 通过环境变量 `QWEN_API_KEY` 设置 API Key

## 注意事项

1. **SHA1 计算**：基于文件内容计算 SHA1，确保唯一性
2. **跳过已处理文件**：`process_documents()` 默认会跳过已处理的文件
3. **异步处理**：所有处理流程都是异步的，需要使用 `await`
4. **错误处理**：每个步骤都有异常处理，失败的文件会记录但不会中断整个流程

## 与 RAG-cy 的兼容性

- Chunk JSON 格式完全兼容 RAG-cy
- FAISS 索引文件格式兼容
- 文件命名方式一致（使用 SHA1）

## 示例：完整处理流程

```python
import asyncio
from pathlib import Path
from app.services.pipeline import RAGPipeline, PipelinePaths

async def process_all_documents():
    # 配置路径
    paths = PipelinePaths(
        base_dir="./data",
        documents_dir="documents",
        markdown_dir="debug_data",
        chunked_reports_dir="metadata/chunked_reports",
        vector_dbs_dir="metadata/vector_dbs"
    )
    
    # 创建 Pipeline
    pipeline = RAGPipeline(paths=paths)
    
    # 处理所有 PDF 文档
    await pipeline.process_documents(skip_existing=True)
    
    print("所有文档处理完成！")

if __name__ == "__main__":
    asyncio.run(process_all_documents())
```
