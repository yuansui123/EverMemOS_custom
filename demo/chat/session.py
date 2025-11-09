"""对话会话管理

管理单个群组的对话会话，提供记忆检索和 LLM 对话功能。
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from agentic_layer.memory_manager import MemoryManager
from demo.memory_config import ChatModeConfig, LLMConfig, ScenarioType
from demo.memory_utils import query_memcells_by_group_and_time
from demo.i18n_texts import I18nTexts
from memory_layer.llm.llm_provider import LLMProvider
from common_utils.datetime_utils import get_now_with_timezone


class ChatSession:
    """对话会话管理器"""
    
    def __init__(
        self,
        group_id: str,
        config: ChatModeConfig,
        llm_config: LLMConfig,
        scenario_type: ScenarioType,
        retrieval_mode: str,  # "rrf" / "embedding" / "bm25"
        data_source: str,     # "memcell" / "event_log"
        texts: I18nTexts,
    ):
        """初始化对话会话
        
        Args:
            group_id: 群组 ID
            config: 对话模式配置
            llm_config: LLM 配置
            scenario_type: 场景类型
            retrieval_mode: 检索模式（rrf/embedding/bm25）
            data_source: 数据源（memcell/event_log）
            texts: 国际化文本对象
        """
        self.group_id = group_id
        self.config = config
        self.llm_config = llm_config
        self.scenario_type = scenario_type
        self.retrieval_mode = retrieval_mode
        self.data_source = data_source
        self.texts = texts
        
        # 会话状态
        self.conversation_history: List[Tuple[str, str]] = []
        self.memcell_count: int = 0
        
        # 服务
        self.llm_provider: Optional[LLMProvider] = None
        self.memory_manager: Optional[MemoryManager] = None
        
        # 最后一次检索元数据
        self.last_retrieval_metadata: Optional[Dict[str, Any]] = None
    
    async def initialize(self) -> bool:
        """初始化会话
        
        Returns:
            初始化是否成功
        """
        try:
            display_name = "group_chat" if self.group_id == "AI产品群" else self.group_id
            print(f"\n[{self.texts.get('loading_label')}] {self.texts.get('loading_group_data', name=display_name)}")
            
            # 统计 MemCell 数量
            now = get_now_with_timezone()
            start_date = now - timedelta(days=self.config.time_range_days)
            memcells = await query_memcells_by_group_and_time(self.group_id, start_date, now)
            self.memcell_count = len(memcells)
            print(f"[{self.texts.get('loading_label')}] {self.texts.get('loading_memories_success', count=self.memcell_count)} ✅")
            
            # 加载对话历史
            loaded_history_count = await self.load_conversation_history()
            if loaded_history_count > 0:
                print(f"[{self.texts.get('loading_label')}] {self.texts.get('loading_history_success', count=loaded_history_count)} ✅")
            else:
                print(f"[{self.texts.get('loading_label')}] {self.texts.get('loading_history_new')} ✅")
            
            # 创建服务
            self.llm_provider = LLMProvider(
                self.llm_config.provider,
                model=self.llm_config.model,
                api_key=self.llm_config.api_key,
                base_url=self.llm_config.base_url,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens,
            )
            
            self.memory_manager = MemoryManager()
            
            print(f"\n[{self.texts.get('hint_label')}] {self.texts.get('loading_help_hint')}\n")
            return True
        
        except Exception as e:
            print(f"\n[{self.texts.get('error_label')}] {self.texts.get('session_init_error', error=str(e))}")
            import traceback
            traceback.print_exc()
            return False
    
    async def load_conversation_history(self) -> int:
        """从文件加载对话历史
        
        Returns:
            加载的对话轮数
        """
        try:
            display_name = "group_chat" if self.group_id == "AI产品群" else self.group_id
            history_files = sorted(
                self.config.chat_history_dir.glob(f"{display_name}_*.json"),
                reverse=True
            )
            
            if not history_files:
                return 0
            
            latest_file = history_files[0]
            with latest_file.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
            
            history = data.get("conversation_history", [])
            self.conversation_history = [
                (item["user_input"], item["assistant_response"])
                for item in history[-self.config.conversation_history_size:]
            ]
            
            return len(self.conversation_history)
        
        except Exception as e:
            print(f"[{self.texts.get('warning_label')}] {self.texts.get('loading_history_new')}: {e}")
            return 0
    
    async def save_conversation_history(self) -> None:
        """保存对话历史到文件"""
        try:
            display_name = "group_chat" if self.group_id == "AI产品群" else self.group_id
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            filename = f"{display_name}_{timestamp}.json"
            filepath = self.config.chat_history_dir / filename
            
            data = {
                "group_id": self.group_id,
                "last_updated": datetime.now().isoformat(),
                "conversation_history": [
                    {
                        "timestamp": datetime.now().isoformat(),
                        "user_input": user_q,
                        "assistant_response": assistant_a,
                    }
                    for user_q, assistant_a in self.conversation_history
                ],
            }
            
            with filepath.open("w", encoding="utf-8") as fp:
                json.dump(data, fp, ensure_ascii=False, indent=2)
            
            print(f"[{self.texts.get('save_label')}] {filename} ✅")
        
        except Exception as e:
            print(f"[{self.texts.get('error_label')}] {e}")
    
    async def retrieve_memories(self, query: str) -> List[Dict[str, Any]]:
        """检索相关记忆 - 支持多种检索模式
        
        Args:
            query: 用户查询
            
        Returns:
            检索到的记忆列表
        """
        if not self.memory_manager:
            raise RuntimeError("请先调用 initialize() 初始化会话")
        
        # 🔥 根据检索模式选择不同的 API
        if self.retrieval_mode == "agentic":
            # Agentic 检索：需要 LLM Provider
            result = await self.memory_manager.retrieve_agentic(
                query=query,
                user_id="default",
                group_id=self.group_id,
                time_range_days=self.config.time_range_days,
                top_k=self.config.top_k_memories,
                llm_provider=self.llm_provider,  # 传递 LLM Provider
                agentic_config=None,  # 使用默认配置
            )
        else:
            # 其他模式：使用 retrieve_lightweight API
            result = await self.memory_manager.retrieve_lightweight(
                query=query,
                user_id="default",
                group_id=self.group_id,
                top_k=self.config.top_k_memories,
                time_range_days=self.config.time_range_days,
                retrieval_mode=self.retrieval_mode,  # rrf / embedding / bm25
                data_source=self.data_source,        # memcell / event_log
            )
        
        # 提取结果和元数据
        memories = result.get("memories", [])
        metadata = result.get("metadata", {})
        
        # 保存元数据（用于 UI 显示）
        self.last_retrieval_metadata = metadata
        
        return memories
    
    def build_prompt(self, user_query: str, memories: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """构建 Prompt
        
        Args:
            user_query: 用户查询
            memories: 检索到的记忆列表
            
        Returns:
            Chat Messages 列表
        """
        messages = []
        
        # System Message
        lang_key = "zh" if self.texts.language == "zh" else "en"
        system_content = self.texts.get(f"prompt_system_role_{lang_key}")
        messages.append({"role": "system", "content": system_content})
        
        # Retrieved Memories
        if memories:
            memory_lines = []
            for i, mem in enumerate(memories, start=1):
                timestamp = mem.get("timestamp", "")[:10]
                subject = mem.get("subject", "")
                summary = mem.get("summary", "")
                episode = mem.get("episode", "")
                
                parts = [f"[{i}] {self.texts.get('prompt_memory_date', date=timestamp)}"]
                if subject:
                    parts.append(self.texts.get("prompt_memory_subject", subject=subject))
                if summary:
                    parts.append(self.texts.get("prompt_memory_content", content=summary))
                if episode:
                    parts.append(self.texts.get("prompt_memory_episode", episode=episode))
                
                memory_lines.append(" | ".join(parts))
            
            memory_content = self.texts.get("prompt_memories_prefix") + "\n".join(memory_lines)
            messages.append({"role": "system", "content": memory_content})
        
        # Conversation History
        for user_q, assistant_a in self.conversation_history[-self.config.conversation_history_size:]:
            messages.append({"role": "user", "content": user_q})
            messages.append({"role": "assistant", "content": assistant_a})
        
        # Current Question
        messages.append({"role": "user", "content": user_query})
        
        return messages
    
    async def chat(self, user_input: str) -> str:
        """核心对话逻辑
        
        Args:
            user_input: 用户输入
            
        Returns:
            助手回答
        """
        from .ui import ChatUI
        
        # 检索记忆
        memories = await self.retrieve_memories(user_input)
        
        # 显示检索结果
        if self.config.show_retrieved_memories and memories:
            ChatUI.print_retrieved_memories(
                memories[:5],
                texts=self.texts,
                retrieval_metadata=self.last_retrieval_metadata,
            )
        
        # 构建 Prompt
        messages = self.build_prompt(user_input, memories)
        
        # 显示生成进度
        ChatUI.print_generating_indicator(self.texts)
        
        # 调用 LLM
        try:
            if hasattr(self.llm_provider, 'provider') and hasattr(
                self.llm_provider.provider, 'chat_with_messages'
            ):
                raw_response = await self.llm_provider.provider.chat_with_messages(messages)
            else:
                prompt_parts = []
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "system":
                        prompt_parts.append(f"System: {content}")
                    elif role == "user":
                        prompt_parts.append(f"User: {content}")
                    elif role == "assistant":
                        prompt_parts.append(f"Assistant: {content}")
                
                prompt = "\n\n".join(prompt_parts)
                raw_response = await self.llm_provider.generate(prompt)
            
            raw_response = raw_response.strip()
            
            # 清除生成进度
            ChatUI.print_generation_complete(self.texts)
            
            assistant_response = raw_response
        
        except Exception as e:
            ChatUI.clear_progress_indicator()
            error_msg = f"[{self.texts.get('error_label')}] {self.texts.get('chat_llm_error', error=str(e))}"
            print(f"\n{error_msg}")
            import traceback
            traceback.print_exc()
            return error_msg
        
        # 更新对话历史
        self.conversation_history.append((user_input, assistant_response))
        
        if len(self.conversation_history) > self.config.conversation_history_size:
            self.conversation_history = self.conversation_history[-self.config.conversation_history_size:]
        
        return assistant_response
    
    def clear_history(self) -> None:
        """清空对话历史"""
        from .ui import ChatUI
        count = len(self.conversation_history)
        self.conversation_history = []
        ChatUI.print_info(self.texts.get("cmd_clear_done", count=count), self.texts)
    
    async def reload_data(self) -> None:
        """重新加载记忆数据"""
        from .ui import ChatUI
        from common_utils.cli_ui import CLIUI
        
        display_name = "group_chat" if self.group_id == "AI产品群" else self.group_id
        
        ui = CLIUI()
        print()
        ui.note(self.texts.get("cmd_reload_refreshing", name=display_name), icon="🔄")
        
        # 重新统计 MemCell 数量
        now = get_now_with_timezone()
        start_date = now - timedelta(days=self.config.time_range_days)
        memcells = await query_memcells_by_group_and_time(self.group_id, start_date, now)
        self.memcell_count = len(memcells)
        
        print()
        ui.success(f"✓ {self.texts.get('cmd_reload_complete', users=0, memories=self.memcell_count)}")
        print()

