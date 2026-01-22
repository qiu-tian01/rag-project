# LangChain 安装指南

## Conda 环境安装

在 `rag-backend` conda 环境中安装 LangChain：

```bash
# 激活 conda 环境
conda activate rag-backend

# 使用 pip 安装 langchain（推荐）
pip install langchain>=0.3.3

# 或者使用 conda 安装（如果可用）
conda install -c conda-forge langchain
```

## 从 requirements.txt 安装

```bash
# 激活 conda 环境
conda activate rag-backend

# 进入 backend 目录
cd backend

# 安装所有依赖（包括 langchain）
pip install -r requirements.txt
```

## 验证安装

安装完成后，可以验证 LangChain 是否正确安装：

```bash
python -c "from langchain.text_splitter import RecursiveCharacterTextSplitter; print('LangChain 安装成功！')"
```

## 注意事项

- LangChain 版本要求：`>=0.3.3`（与 RAG-cy 项目保持一致）
- 确保已安装 `tiktoken`，因为 `RecursiveCharacterTextSplitter.from_tiktoken_encoder()` 需要它
- 如果遇到导入错误，请确保在正确的 conda 环境中运行
