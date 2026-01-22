# RAG 后端服务

基于 FastAPI 的 RAG（检索增强生成）后端服务，用于需求文档检索。

## 环境要求

- Python 3.9+
- pip 或 conda

## 快速开始

### 1. 创建虚拟环境（推荐）

```bash
# 使用 venv
python -m venv venv

# Windows 激活虚拟环境
venv\Scripts\activate

# Linux/Mac 激活虚拟环境
source venv/bin/activate
```

### 2. 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

### 3. 配置环境变量

复制环境变量模板文件：

```bash
# Windows
copy env.example .env

# Linux/Mac
cp env.example .env
```

编辑 `.env` 文件，填入实际的 API 密钥：

```env
# Qwen API 配置（必需）
QWEN_API_KEY=your_qwen_api_key
QWEN_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-plus

# Embedding 配置
EMBEDDING_MODEL=qwen-embedding-v3

# Jina Reranker 配置（必需）
JINA_API_KEY=your_jina_api_key
JINA_API_BASE_URL=https://api.jina.ai/v1/rerank
JINA_RERANK_MODEL=jina-reranker-v2-base-multilingual

# FAISS 配置
FAISS_INDEX_PATH=./data/index/faiss.index
METADATA_PATH=./data/metadata/chunks.json

# 文档路径
DOCUMENTS_PATH=./data/documents
```

**重要**：请确保至少配置了 `QWEN_API_KEY` 和 `JINA_API_KEY`，否则服务可能无法正常启动。

### 4. 启动服务

#### 方式一：使用 uvicorn 命令（推荐）

```bash
# 开发模式（自动重载）
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 生产模式
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

#### 方式二：使用 Python 直接运行

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. 验证服务

服务启动后，访问以下地址：

- **API 文档**：http://localhost:8000/docs
- **健康检查**：http://localhost:8000/health
- **根路径**：http://localhost:8000/

## API 接口

### 健康检查

```bash
GET /health
```

### 检索接口

```bash
POST /api/v1/search
Content-Type: application/json

{
  "query": "查询文本",
  "top_k": 10
}
```

## 项目结构

```
backend/
├── app/
│   ├── main.py              # FastAPI 应用入口
│   ├── api/                 # API 路由
│   │   └── v1/
│   │       └── search.py    # 检索 API
│   ├── services/            # 业务服务
│   │   ├── embedding.py     # Embedding 服务
│   │   ├── rerank.py        # 重排服务
│   │   ├── llm.py           # LLM 服务
│   │   └── retrieval.py     # 检索服务
│   ├── models/              # 数据模型
│   ├── storage/             # 存储服务
│   └── utils/               # 工具函数
├── data/                    # 数据目录
│   ├── documents/           # 文档存储
│   ├── index/               # FAISS 索引
│   └── metadata/            # 元数据
├── requirements.txt         # 依赖列表
├── env.example              # 环境变量模板
└── README.md                # 本文档
```

## 开发说明

### 开发模式

使用 `--reload` 参数启动开发模式，代码修改后会自动重载：

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 日志

默认日志会输出到控制台。生产环境建议配置日志文件：

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

## 常见问题

### 1. 端口被占用

如果 8000 端口被占用，可以指定其他端口：

```bash
uvicorn app.main:app --reload --port 8001
```

### 2. 模块导入错误

确保在 `backend` 目录下运行命令，或者使用绝对路径：

```bash
# 在 backend 目录下
uvicorn app.main:app --reload

# 或者在项目根目录下
uvicorn backend.app.main:app --reload
```

### 3. 环境变量未加载

确保 `.env` 文件在 `backend` 目录下，并且安装了 `python-dotenv`：

```bash
pip install python-dotenv
```

### 4. API 密钥错误

检查 `.env` 文件中的 API 密钥是否正确，确保：
- `QWEN_API_KEY` 已配置
- `JINA_API_KEY` 已配置

## 生产部署

### 使用 Gunicorn（推荐）

```bash
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### 使用 Docker

创建 `Dockerfile`：

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

构建和运行：

```bash
docker build -t rag-backend .
docker run -p 8000:8000 --env-file .env rag-backend
```

## 许可证

MIT


