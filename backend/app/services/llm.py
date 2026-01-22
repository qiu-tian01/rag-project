"""
LLM 服务
使用 Qwen API 进行文本生成，支持多模型选择
"""
import logging
import time
import os
from typing import Optional, List, Dict
import dashscope
from dashscope import AioGeneration
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# 模型映射：1=qwen-max, 2=qwen-plus, 3=qwen-turbo
MODEL_MAP = {
    1: "qwen-max",
    2: "qwen-plus",
    3: "qwen-turbo"
}

class LLMService:
    """LLM 服务类，使用 DashScope API 调用通义千问模型（支持异步和多模型选择）"""
    
    def __init__(self, model_id: Optional[int] = None):
        """
        初始化LLM服务
        
        Args:
            model_id: 模型ID (1=qwen-max, 2=qwen-plus, 3=qwen-turbo)，如果为None则从环境变量读取
        """
        self.api_key = os.getenv("QWEN_API_KEY")
        if not self.api_key:
            raise ValueError("未找到 QWEN_API_KEY 环境变量")
        dashscope.api_key = self.api_key
        
        if model_id is not None:
            self.model = self._get_model_name(model_id)
        else:
            env_model = os.getenv("QWEN_MODEL", "qwen-plus")
            # 如果环境变量是模型ID（数字字符串），转换为模型名
            if env_model.isdigit():
                self.model = self._get_model_name(int(env_model))
            else:
                self.model = env_model
    
    def _get_model_name(self, model_id: int) -> str:
        """
        根据模型ID获取模型名称
        
        Args:
            model_id: 模型ID (1, 2, 或 3)
            
        Returns:
            模型名称字符串
            
        Raises:
            ValueError: 如果model_id无效
        """
        if model_id not in MODEL_MAP:
            raise ValueError(f"无效的模型ID: {model_id}，支持的值: 1=qwen-max, 2=qwen-plus, 3=qwen-turbo")
        return MODEL_MAP[model_id]
    
    def set_model(self, model_id: int):
        """
        设置当前使用的模型
        
        Args:
            model_id: 模型ID (1=qwen-max, 2=qwen-plus, 3=qwen-turbo)
        """
        old_model = self.model
        self.model = self._get_model_name(model_id)
        logger.info(f"[LLM] 模型切换: {old_model} -> {self.model}")
    
    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = "你是一个专业的需求分析助手，请根据提供的上下文回答用户的问题。",
        model: Optional[str] = None
    ) -> str:
        """
        生成文本 (异步)
        
        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            model: 指定使用的模型名称，如果为None则使用实例的model属性
        """
        generate_start = time.time()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # 使用指定的模型或实例的模型
        model_to_use = model or self.model
        prompt_length = len(prompt)
        system_length = len(system_prompt) if system_prompt else 0
        
        logger.info(f"[LLM] 调用生成接口 (模型: {model_to_use}, prompt长度: {prompt_length}, system长度: {system_length})")
        
        try:
            response = await AioGeneration.call(
                model=model_to_use,
                messages=messages,
                result_format='message'
            )
            
            elapsed_time = time.time() - generate_start
            
            if response.status_code == 200:
                content = response.output.choices[0].message.content
                logger.info(f"[LLM] 生成成功 (耗时: {elapsed_time:.2f}秒, 输出长度: {len(content)} 字符)")
                logger.debug(f"[LLM] 输出预览: {content[:200]}...")
                return content
            else:
                logger.error(f"[LLM] 调用失败: {response.code} - {response.message} (耗时: {elapsed_time:.2f}秒)")
                raise Exception(f"调用 LLM 失败: {response.code} - {response.message}")
        except Exception as e:
            elapsed_time = time.time() - generate_start
            logger.error(f"[LLM] 生成异常: {str(e)} (耗时: {elapsed_time:.2f}秒)", exc_info=True)
            raise
    
    async def rewrite_query(self, query: str, model: Optional[str] = None) -> str:
        """
        查询改写 (异步)
        
        Args:
            query: 原始查询
            model: 指定使用的模型，如果为None则使用实例的model属性
        """
        rewrite_start = time.time()
        logger.debug(f"[LLM] 开始查询改写 (原始查询: {query[:100]}...)")
        
        rewrite_prompt = f"请将以下用户查询改写为更适合语义搜索的关键词或描述性语句，只需返回改写后的内容：\n\n查询：{query}"
        result = await self.generate(rewrite_prompt, system_prompt="你是一个搜索优化专家。", model=model)
        
        elapsed_time = time.time() - rewrite_start
        logger.debug(f"[LLM] 查询改写完成 (耗时: {elapsed_time:.2f}秒, 改写后: {result[:100]}...)")
        
        return result

