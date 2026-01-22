# 文档处理 API 使用指南

本文档说明如何使用文档处理 API 接口，实现 PDF 文件的上传、处理和向量数据库存储。

## API 接口列表

### 1. 上传并处理文档

**接口地址**: `POST /api/v1/documents/upload`

**功能**: 上传 PDF 文件并自动完成处理流程（转换 → Chunk → Embedding → 存储向量数据库）

**请求方式**: `multipart/form-data`

**请求参数**:
- `file` (File, 必需): PDF 文件
- `company_name` (String, 可选): 公司名称，用于元数据
- `chunk_size` (Integer, 可选): 每个 chunk 的最大行数，默认 30
- `chunk_overlap` (Integer, 可选): chunk 之间的重叠行数，默认 5

**响应示例**:
```json
{
  "success": true,
  "message": "文档处理成功",
  "file_name": "document.pdf",
  "sha1": "abc123def456...",
  "markdown_path": "data/debug_data/document.md",
  "chunk_json_path": "data/metadata/chunked_reports/document.json",
  "faiss_path": "data/metadata/vector_dbs/abc123def456.faiss",
  "chunk_count": 45
}
```

### 2. 处理已存在的文档

**接口地址**: `POST /api/v1/documents/process`

**功能**: 处理服务器上已存在的 PDF 文件

**请求方式**: `application/x-www-form-urlencoded` 或 `multipart/form-data`

**请求参数**:
- `file_path` (String, 必需): PDF 文件路径（相对于 documents 目录）
- `company_name` (String, 可选): 公司名称
- `chunk_size` (Integer, 可选): 每个 chunk 的最大行数，默认 30
- `chunk_overlap` (Integer, 可选): chunk 之间的重叠行数，默认 5

**响应格式**: 同上传接口

### 3. 列出所有文档

**接口地址**: `GET /api/v1/documents/list`

**功能**: 获取所有已处理的文档列表

**响应示例**:
```json
{
  "total": 3,
  "documents": [
    {
      "file_name": "document1.pdf",
      "file_path": "document1.pdf",
      "file_size": 2048576,
      "sha1": "abc123...",
      "processed": true,
      "chunk_json_exists": true,
      "faiss_exists": true
    }
  ]
}
```

## 使用示例

### Python 示例

```python
import requests

# 1. 上传并处理文档
url = "http://localhost:8000/api/v1/documents/upload"

with open("document.pdf", "rb") as f:
    files = {"file": ("document.pdf", f, "application/pdf")}
    data = {
        "company_name": "示例公司",
        "chunk_size": 30,
        "chunk_overlap": 5
    }
    response = requests.post(url, files=files, data=data)
    result = response.json()
    print(result)

# 2. 处理已存在的文档
url = "http://localhost:8000/api/v1/documents/process"
data = {
    "file_path": "document.pdf",
    "company_name": "示例公司"
}
response = requests.post(url, data=data)
result = response.json()
print(result)

# 3. 列出所有文档
url = "http://localhost:8000/api/v1/documents/list"
response = requests.get(url)
result = response.json()
print(result)
```

### JavaScript/TypeScript 示例

```typescript
// 1. 上传并处理文档
async function uploadAndProcess(file: File, companyName?: string) {
  const formData = new FormData();
  formData.append('file', file);
  if (companyName) {
    formData.append('company_name', companyName);
  }
  formData.append('chunk_size', '30');
  formData.append('chunk_overlap', '5');

  const response = await fetch('http://localhost:8000/api/v1/documents/upload', {
    method: 'POST',
    body: formData
  });

  const result = await response.json();
  return result;
}

// 2. 处理已存在的文档
async function processExisting(filePath: string, companyName?: string) {
  const formData = new FormData();
  formData.append('file_path', filePath);
  if (companyName) {
    formData.append('company_name', companyName);
  }

  const response = await fetch('http://localhost:8000/api/v1/documents/process', {
    method: 'POST',
    body: formData
  });

  const result = await response.json();
  return result;
}

// 3. 列出所有文档
async function listDocuments() {
  const response = await fetch('http://localhost:8000/api/v1/documents/list');
  const result = await response.json();
  return result;
}
```

### cURL 示例

```bash
# 1. 上传并处理文档
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@document.pdf" \
  -F "company_name=示例公司" \
  -F "chunk_size=30" \
  -F "chunk_overlap=5"

# 2. 处理已存在的文档
curl -X POST "http://localhost:8000/api/v1/documents/process" \
  -F "file_path=document.pdf" \
  -F "company_name=示例公司"

# 3. 列出所有文档
curl -X GET "http://localhost:8000/api/v1/documents/list"
```

## 处理流程说明

当调用上传或处理接口时，系统会自动执行以下步骤：

1. **PDF → Markdown 转换**
   - 使用 `PDFToMarkdownService` 将 PDF 转换为 Markdown
   - 保存到 `data/debug_data/` 目录

2. **Markdown → Chunks 切分**
   - 使用 `DocumentChunker` 将 Markdown 切分为 chunks
   - 保存为 JSON 格式到 `data/metadata/chunked_reports/`
   - JSON 格式包含 `metainfo` 和 `content.chunks`

3. **Chunks → Embeddings 生成**
   - 使用 `EmbeddingService` 为每个 chunk 生成向量
   - 调用 DashScope API 生成 embeddings

4. **创建 FAISS 向量索引**
   - 使用 `VectorDBService` 创建 FAISS 索引
   - 保存到 `data/metadata/vector_dbs/`
   - 文件名使用 SHA1 哈希值

## 错误处理

### 常见错误码

- `400 Bad Request`: 请求参数错误（如文件格式不正确、文件为空等）
- `404 Not Found`: 文件不存在（处理已存在文档时）
- `500 Internal Server Error`: 处理失败（如 API 调用失败、文件损坏等）

### 错误响应格式

```json
{
  "detail": "错误描述信息"
}
```

## 注意事项

1. **文件大小限制**: 建议单个 PDF 文件不超过 100MB
2. **处理时间**: 大文件处理可能需要较长时间，建议前端实现进度提示
3. **API 限流**: Embedding 生成受 API 限流影响，大量文件处理时注意控制并发
4. **文件存储**: 上传的文件会保存在 `data/documents/` 目录
5. **重复处理**: 相同文件（相同 SHA1）会被覆盖，不会创建重复索引

## 前端集成建议

1. **文件上传**: 使用 `<input type="file">` 或文件上传组件
2. **进度显示**: 可以轮询处理状态或使用 WebSocket 获取实时进度
3. **错误处理**: 捕获并显示 API 返回的错误信息
4. **成功提示**: 显示处理结果，包括生成的文件路径和统计信息

## 测试

启动服务后，可以访问 Swagger UI 进行测试：

```
http://localhost:8000/docs
```

在 Swagger UI 中可以：
- 查看所有 API 接口
- 测试接口功能
- 查看请求和响应格式
