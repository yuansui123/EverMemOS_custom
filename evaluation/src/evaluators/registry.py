"""
Evaluator 注册机制

提供评估器的注册和创建功能。
使用 lazy loading 策略，保持 __init__.py 为空。
"""
import importlib
from typing import Dict, Type, List
from evaluation.src.evaluators.base import BaseEvaluator


_EVALUATOR_REGISTRY: Dict[str, Type[BaseEvaluator]] = {}

# 评估器模块映射（用于延迟加载）
_EVALUATOR_MODULES = {
    "llm_judge": "evaluation.src.evaluators.llm_judge",
    "exact_match": "evaluation.src.evaluators.exact_match",
    # 未来添加其他评估器：
    # "bert_score": "evaluation.src.evaluators.bert_score",
}


def register_evaluator(name: str):
    """
    注册评估器的装饰器
    
    Usage:
        @register_evaluator("llm_judge")
        class LLMJudge(BaseEvaluator):
            ...
    """
    def decorator(cls: Type[BaseEvaluator]):
        _EVALUATOR_REGISTRY[name] = cls
        return cls
    return decorator


def _ensure_evaluator_loaded(name: str):
    """
    确保指定的评估器已加载（延迟加载策略）
    
    通过动态导入模块来触发 @register_evaluator 装饰器的执行。
    这样可以保持 __init__.py 为空，符合项目规范。
    
    Args:
        name: 评估器名称
        
    Raises:
        ValueError: 如果评估器不存在
        RuntimeError: 如果模块加载后仍未注册
    """
    if name in _EVALUATOR_REGISTRY:
        return  # 已加载
    
    if name not in _EVALUATOR_MODULES:
        raise ValueError(
            f"Unknown evaluator: {name}. "
            f"Available evaluators: {list(_EVALUATOR_MODULES.keys())}"
        )
    
    # 动态导入模块，触发 @register_evaluator 装饰器执行
    module_path = _EVALUATOR_MODULES[name]
    importlib.import_module(module_path)
    
    # 验证注册是否成功
    if name not in _EVALUATOR_REGISTRY:
        raise RuntimeError(
            f"Evaluator '{name}' module loaded but not registered. "
            f"Check if @register_evaluator('{name}') decorator is present."
        )


def create_evaluator(name: str, llm_provider=None) -> BaseEvaluator:
    """
    创建评估器实例
    
    Args:
        name: 评估器名称
        llm_provider: LLM 提供者（某些评估器需要）
        
    Returns:
        评估器实例
        
    Raises:
        ValueError: 如果评估器未注册
    """
    # 延迟加载：确保评估器已加载
    _ensure_evaluator_loaded(name)
    
    return _EVALUATOR_REGISTRY[name](llm_provider)


def list_evaluators() -> List[str]:
    """列出所有可用的评估器"""
    return list(_EVALUATOR_MODULES.keys())
