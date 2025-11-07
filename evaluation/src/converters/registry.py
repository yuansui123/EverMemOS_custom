"""
Converter 注册机制

提供转换器的注册和创建功能。
使用 lazy loading 策略，保持 __init__.py 为空。
"""
import importlib
from typing import Dict, Type, List, Optional
from evaluation.src.converters.base import BaseConverter


_CONVERTER_REGISTRY: Dict[str, Type[BaseConverter]] = {}

# 转换器模块映射（用于延迟加载）
_CONVERTER_MODULES = {
    "longmemeval": "evaluation.src.converters.longmemeval_converter",
    "personamem": "evaluation.src.converters.personamem_converter",
    # 未来添加其他转换器
}


def register_converter(name: str):
    """
    注册转换器的装饰器
    
    Usage:
        @register_converter("longmemeval")
        class LongMemEvalConverter(BaseConverter):
            ...
    """
    def decorator(cls: Type[BaseConverter]):
        _CONVERTER_REGISTRY[name] = cls
        return cls
    return decorator


def _ensure_converter_loaded(name: str):
    """
    确保指定的转换器已加载（延迟加载策略）
    
    Args:
        name: 转换器名称
        
    Raises:
        ValueError: 如果转换器不存在
        RuntimeError: 如果模块加载后仍未注册
    """
    if name in _CONVERTER_REGISTRY:
        return  # 已加载
    
    if name not in _CONVERTER_MODULES:
        raise ValueError(
            f"Unknown converter: {name}. "
            f"Available converters: {list(_CONVERTER_MODULES.keys())}"
        )
    
    # 动态导入模块，触发 @register_converter 装饰器执行
    module_path = _CONVERTER_MODULES[name]
    importlib.import_module(module_path)
    
    # 验证注册是否成功
    if name not in _CONVERTER_REGISTRY:
        raise RuntimeError(
            f"Converter '{name}' module loaded but not registered. "
            f"Check if @register_converter('{name}') decorator is present."
        )


def get_converter(name: str) -> Optional[BaseConverter]:
    """
    获取转换器实例（如果存在）
    
    Args:
        name: 转换器名称
        
    Returns:
        转换器实例，如果不存在则返回 None
    """
    if name not in _CONVERTER_MODULES:
        return None  # 该数据集不需要转换
    
    # 延迟加载：确保转换器已加载
    _ensure_converter_loaded(name)
    
    return _CONVERTER_REGISTRY[name]()


def list_converters() -> List[str]:
    """列出所有可用的转换器"""
    return list(_CONVERTER_MODULES.keys())


