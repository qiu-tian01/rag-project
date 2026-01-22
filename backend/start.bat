@echo off
REM Windows 启动脚本
echo 正在启动 RAG 后端服务...
echo.

REM 检查虚拟环境
if exist "venv\Scripts\activate.bat" (
    echo 激活虚拟环境...
    call venv\Scripts\activate.bat
)

REM 检查 .env 文件
if not exist ".env" (
    echo 警告: .env 文件不存在，请从 env.example 复制并配置
    echo.
)

REM 启动服务
echo 启动 FastAPI 服务...
echo 访问地址: http://localhost:8000
echo API 文档: http://localhost:8000/docs
echo.
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

pause


