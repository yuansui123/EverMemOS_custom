"""
在线 API Adapter 基类

为所有在线记忆系统 API（Mem0, Memos, Memu 等）提供通用功能。
所有在线 API adapter 可以继承此类。

设计理念：
- 提供默认的 answer() 实现（使用通用 prompt）
- 子类可以重写 answer() 使用自己的特定 prompt
- 提供辅助方法用于数据格式转换
"""
from abc import abstractmethod
from pathlib import Path
from typing import Any, List, Dict, Optional

from evaluation.src.adapters.base import BaseAdapter
from evaluation.src.core.data_models import Conversation, SearchResult
from evaluation.src.utils.config import load_yaml

# 导入 Memory Layer 组件
from memory_layer.llm.llm_provider import LLMProvider


class OnlineAPIAdapter(BaseAdapter):
    """
    在线 API Adapter 基类
    
    提供通用功能：
    1. LLM Provider 初始化
    2. Answer 生成（复用 EverMemOS 的实现）
    3. 标准格式转换辅助方法
    
    子类只需实现：
    - add(): 调用在线 API 摄入数据
    - search(): 调用在线 API 检索
    """
    
    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config)
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        
        # 初始化 LLM Provider（用于 answer 生成）
        llm_config = config.get("llm", {})
        
        self.llm_provider = LLMProvider(
            provider_type=llm_config.get("provider", "openai"),
            model=llm_config.get("model", "gpt-4o-mini"),
            api_key=llm_config.get("api_key", ""),
            base_url=llm_config.get("base_url", "https://api.openai.com/v1"),
            temperature=llm_config.get("temperature", 0.3),
            max_tokens=llm_config.get("max_tokens", 32768),
        )
        
        # 加载 prompts（从 YAML 文件）
        evaluation_root = Path(__file__).parent.parent.parent
        prompts_path = evaluation_root / "config" / "prompts.yaml"
        self._prompts = load_yaml(str(prompts_path))
        
        print(f"✅ {self.__class__.__name__} initialized")
        print(f"   LLM Model: {llm_config.get('model')}")
        print(f"   Output Dir: {self.output_dir}")
    
    @abstractmethod
    async def add(
        self, 
        conversations: List[Conversation],
        **kwargs
    ) -> Dict[str, Any]:
        """
        摄入对话数据（调用在线 API）
        
        子类必须实现此方法。
        """
        pass
    
    @abstractmethod
    async def search(
        self, 
        query: str,
        conversation_id: str,
        index: Any,
        **kwargs
    ) -> SearchResult:
        """
        检索相关记忆（调用在线 API）
        
        子类必须实现此方法。
        """
        pass
    
    async def answer(self, query: str, context: str, **kwargs) -> str:
        """
        生成答案（使用通用 MEMOS prompt）
        
        子类可以重写此方法以使用自己特定的 prompt。
        默认使用 ANSWER_PROMPT_MEMOS（适用于大多数系统）。
        """
        # 获取 answer prompt（子类可以重写 _get_answer_prompt）
        prompt = self._get_answer_prompt().format(context=context, question=query)
        
        # 获取重试次数
        max_retries = self.config.get("answer", {}).get("max_retries", 3)
        
        # 生成答案
        for i in range(max_retries):
            try:
                result = await self.llm_provider.generate(
                    prompt=prompt,
                    temperature=0,
                    max_tokens=32768,
                )
                
                # 清理结果（移除可能的 "FINAL ANSWER:" 前缀）
                if "FINAL ANSWER:" in result:
                    parts = result.split("FINAL ANSWER:")
                    if len(parts) > 1:
                        result = parts[1].strip()
                    else:
                        result = result.strip()
                else:
                    result = result.strip()
                
                if result == "":
                    continue
                
                return result
            except Exception as e:
                print(f"⚠️  Answer generation error (attempt {i+1}/{max_retries}): {e}")
                if i == max_retries - 1:
                    raise
                continue
        
        return ""
    
    def _get_answer_prompt(self) -> str:
        """
        获取 answer prompt
        
        子类可以重写此方法返回自己的 prompt。
        默认返回通用 default prompt
        """
        return self._prompts["online_api"]["default"]["answer_prompt"]
    
    # ===== 辅助方法：格式转换 =====
    
    def _conversation_to_messages(
        self, 
        conversation: Conversation,
        format_type: str = "basic",
        perspective: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        将标准 Conversation 转换为消息列表
        
        Args:
            conversation: 标准对话对象
            format_type: 格式类型（basic, mem0, memos, memu）
            perspective: 视角（speaker_a 或 speaker_b），用于双视角系统如Memos
        
        Returns:
            消息列表
        """
        messages = []
        speaker_a = conversation.metadata.get("speaker_a", "")
        speaker_b = conversation.metadata.get("speaker_b", "")
        
        for msg in conversation.messages:
            # 智能判断 role 和 content
            role, content = self._determine_role_and_content(
                msg.speaker_name, 
                msg.content,
                speaker_a,
                speaker_b,
                perspective
            )
            
            # 基础消息
            message = {
                "role": role,
                "content": content,
            }
            
            # 根据不同系统的需求添加额外字段
            if format_type == "mem0":
                # Mem0 格式：需要 timestamp
                if msg.timestamp:
                    from common_utils.datetime_utils import to_iso_format
                    message["timestamp"] = to_iso_format(msg.timestamp)
            
            elif format_type == "memos":
                # Memos 格式：需要 chat_time
                if msg.timestamp:
                    from common_utils.datetime_utils import to_iso_format
                    message["chat_time"] = to_iso_format(msg.timestamp)
            
            elif format_type == "memu":
                # Memu 格式：需要 created_at
                if msg.timestamp:
                    from common_utils.datetime_utils import to_iso_format
                    message["created_at"] = to_iso_format(msg.timestamp)
            
            messages.append(message)
        
        return messages
    
    def _determine_role_and_content(
        self,
        speaker_name: str,
        content: str,
        speaker_a: str,
        speaker_b: str,
        perspective: Optional[str] = None
    ) -> tuple:
        """
        智能判断消息的 role 和 content
        
        对于只支持 user/assistant 的系统（如 Memos），需要特殊处理：
        1. 如果 speaker 是标准角色（user/assistant 及其变体），直接使用
        2. 如果是自定义名称，根据 perspective 转换：
           - 从 speaker_a 视角：speaker_a 的消息是 "user"，speaker_b 是 "assistant"
           - 从 speaker_b 视角：speaker_b 的消息是 "user"，speaker_a 是 "assistant"
        3. Content 对于自定义 speaker，需要加上 "speaker: " 前缀
        
        Args:
            speaker_name: 说话者名称
            content: 消息内容
            speaker_a: 对话中的 speaker_a
            speaker_b: 对话中的 speaker_b
            perspective: 视角（用于双视角系统）
        
        Returns:
            (role, content) 元组
        """
        # 情况1: 标准角色（user/assistant 及其变体）
        speaker_lower = speaker_name.lower()
        
        # 检查是否是标准角色或其变体
        if speaker_lower in ["user", "assistant"]:
            # 完全匹配: "user", "User", "assistant", "Assistant"
            return speaker_lower, content
        elif speaker_lower.startswith("user"):
            # 变体: "user_123", "User_456" 等
            return "user", content
        elif speaker_lower.startswith("assistant"):
            # 变体: "assistant_123", "Assistant_456" 等
            return "assistant", content
        
        # 情况2: 自定义 speaker，需要转换
        # 默认行为：speaker_a 是 user，speaker_b 是 assistant
        if perspective == "speaker_b":
            # 从 speaker_b 的视角
            if speaker_name == speaker_b:
                role = "user"
            elif speaker_name == speaker_a:
                role = "assistant"
            else:
                # 未知 speaker，默认为 assistant
                role = "assistant"
        else:
            # 从 speaker_a 的视角（默认）
            if speaker_name == speaker_a:
                role = "user"
            elif speaker_name == speaker_b:
                role = "assistant"
            else:
                # 未知 speaker，默认为 user
                role = "user"
        
        # 对于自定义 speaker，content 需要加上前缀
        formatted_content = f"{speaker_name}: {content}"
        
        return role, formatted_content
    
    def _extract_user_id(self, conversation: Conversation, speaker: str = "speaker_a") -> str:
        """
        从 Conversation 中提取 user_id（用于在线 API）
        
        逻辑：结合 conversation_id 和 speaker 名字，确保会话隔离
        
        Args:
            conversation: 标准对话对象
            speaker: 说话者标识（speaker_a 或 speaker_b）
        
        Returns:
            user_id 字符串，格式：{conv_id}_{speaker_name}
        
        示例：
            - LoCoMo: speaker_a="Caroline" → user_id="locomo_0_Caroline"
            - PersonaMem: speaker_a="Kanoa Manu" → user_id="personamem_0_Kanoa_Manu"
            - 无 speaker: → user_id="locomo_0_speaker_a"
        
        设计原因：
            - 包含 conv_id：确保不同会话的记忆隔离（评测准确性）
            - 包含 speaker 名字：后台查看更直观（如 Caroline vs speaker_a）
            - 空格转下划线：避免 user_id 中出现空格
        """
        conv_id = conversation.conversation_id
        speaker_name = conversation.metadata.get(speaker)
        
        if speaker_name:
            # 有 speaker 名字：将空格替换为下划线
            speaker_name_normalized = speaker_name.replace(" ", "_")
            user_id = f"{conv_id}_{speaker_name_normalized}"
        else:
            # 没有 speaker 名字：locomo_0_speaker_a
            user_id = f"{conv_id}_{speaker}"
        
        return user_id
    
    def _get_user_id_from_conversation_id(self, conversation_id: str) -> str:
        """
        从 conversation_id 推导 user_id（简化版）
        
        Args:
            conversation_id: 对话 ID
        
        Returns:
            user_id 字符串
        """
        # 简化实现：直接使用 conversation_id
        # 实际使用时可能需要更复杂的映射逻辑
        return conversation_id
    
    def get_system_info(self) -> Dict[str, Any]:
        """返回系统信息"""
        return {
            "name": self.__class__.__name__,
            "type": "online_api",
            "description": f"{self.__class__.__name__} adapter for online memory API",
        }

