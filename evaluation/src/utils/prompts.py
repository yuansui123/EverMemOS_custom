"""
Prompt Utilities

提供 prompt 加载和格式化功能。
"""
from pathlib import Path
from typing import Dict, Any
import yaml


class PromptManager:
    """Prompt 管理器"""
    
    _instance = None
    _prompts = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._prompts is None:
            self._load_prompts()
    
    def _load_prompts(self):
        """加载 prompts 配置文件"""
        # 找到 config/prompts.yaml
        current_file = Path(__file__)
        config_path = current_file.parent.parent.parent / "config" / "prompts.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Prompts config not found: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            self._prompts = yaml.safe_load(f)
    
    def get_prompt(self, prompt_key: str, sub_key: str = None) -> str:
        """
        获取 prompt 模板
        
        Args:
            prompt_key: Prompt 类别键（如 "answer_generation", "llm_judge"）
            sub_key: 子键（如 "system_prompt", "user_prompt"）
            
        Returns:
            Prompt 模板字符串
        
        Example:
            >>> pm = PromptManager()
            >>> pm.get_prompt("answer_generation", "template")
            'Based on the following memories...'
            >>> pm.get_prompt("llm_judge", "system_prompt")
            'You are an expert grader...'
        """
        if prompt_key not in self._prompts:
            raise KeyError(f"Prompt key '{prompt_key}' not found in prompts.yaml")
        
        prompt_config = self._prompts[prompt_key]
        
        if sub_key:
            if sub_key not in prompt_config:
                raise KeyError(
                    f"Sub-key '{sub_key}' not found in prompt '{prompt_key}'"
                )
            return prompt_config[sub_key].strip()
        
        # 如果没有 sub_key，默认返回 'template'
        if "template" in prompt_config:
            return prompt_config["template"].strip()
        
        raise KeyError(
            f"No 'template' field found in prompt '{prompt_key}' "
            f"and no sub_key specified"
        )
    
    def format_prompt(
        self, 
        prompt_key: str, 
        sub_key: str = None, 
        **kwargs
    ) -> str:
        """
        获取并格式化 prompt
        
        Args:
            prompt_key: Prompt 类别键
            sub_key: 子键
            **kwargs: 格式化参数
            
        Returns:
            格式化后的 prompt
        
        Example:
            >>> pm = PromptManager()
            >>> pm.format_prompt(
            ...     "answer_generation",
            ...     context="Memory 1...",
            ...     question="What is X?"
            ... )
            'Based on the following memories...Memory 1...Question: What is X?'
        """
        template = self.get_prompt(prompt_key, sub_key)
        return template.format(**kwargs)


# 全局实例
_prompt_manager = None


def get_prompt_manager() -> PromptManager:
    """获取全局 PromptManager 实例"""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


def get_prompt(prompt_key: str, sub_key: str = None) -> str:
    """快捷方法：获取 prompt"""
    return get_prompt_manager().get_prompt(prompt_key, sub_key)


def format_prompt(prompt_key: str, sub_key: str = None, **kwargs) -> str:
    """快捷方法：格式化 prompt"""
    return get_prompt_manager().format_prompt(prompt_key, sub_key, **kwargs)


