from abc import ABC, abstractmethod
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()


class MemoryProviderBase(ABC):
    def __init__(self, config):
        self.config = config
        self.role_id = None

    def set_llm(self, llm):
        self.llm = llm

    @abstractmethod
    async def save_memory(self, msgs, session_id=None):
        """Save a new memory for specific role and return memory ID"""
        print("this is base func", msgs)

    @abstractmethod
    async def query_memory(self, query: str) -> str:
        """Query memories for specific role based on similarity"""
        return "please implement query method"

    async def build_memory_context(self, query: str, dialogue_messages=None, session_id=None) -> str:
        """
        可选的检索拦截器入口。

        默认实现保持向后兼容，仍然只基于 query 检索。
        新的多维记忆引擎可以覆写该方法，利用当前会话 working memory、
        长期 factual / episodic / semantic memories 一起构造 prompt 上下文。
        """
        return await self.query_memory(query)

    def init_memory(self, role_id, llm, **kwargs):
        self.role_id = role_id
        self.llm = llm
