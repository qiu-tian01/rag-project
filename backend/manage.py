import asyncio
import click
import os
from app.services.pipeline import RAGPipeline

@click.group()
def cli():
    """RAG 系统管理工具"""
    pass

@cli.command()
@click.option('--dir', default='./data/documents', help='文档目录路径')
def ingest(dir):
    """批量入库文档"""
    pipeline = RAGPipeline()
    asyncio.run(pipeline.ingest_directory(dir))

@cli.command()
@click.argument('query')
def query(query):
    """测试单次问答"""
    pipeline = RAGPipeline()
    result = asyncio.run(pipeline.answer(query))
    print(f"\n思考过程:\n{result['thoughts']}")
    print(f"\n最终答案:\n{result['answer']}")
    print(f"\n引用页码: {result['citations']}")

if __name__ == '__main__':
    cli()