"""
Adapter 注册机制

提供适配器的注册和创建功能。
使用 lazy loading 策略，保持 __init__.py 为空。
"""
import importlib
from typing import Dict, Type, List
from evaluation.src.adapters.base import BaseAdapter


_ADAPTER_REGISTRY: Dict[str, Type[BaseAdapter]] = {}

# 适配器模块映射（用于延迟加载）
_ADAPTER_MODULES = {
    # 本地系统
    "evermemos": "evaluation.src.adapters.evermemos_adapter",
    
    # 在线 API 系统
    "mem0": "evaluation.src.adapters.mem0_adapter",
    "memos": "evaluation.src.adapters.memos_adapter",
    "memu": "evaluation.src.adapters.memu_adapter",
    "zep": "evaluation.src.adapters.zep_adapter",
    "memobase": "evaluation.src.adapters.memobase_adapter",
    "supermemory": "evaluation.src.adapters.supermemory_adapter",
    
    # 未来添加其他系统：
    # "nemori": "evaluation.src.adapters.nemori_adapter",
}


def register_adapter(name: str):
    """
    注册适配器的装饰器
    
    Usage:
        @register_adapter("evermemos")
        class EverMemOSAdapter(BaseAdapter):
            ...
    """
    def decorator(cls: Type[BaseAdapter]):
        _ADAPTER_REGISTRY[name] = cls
        return cls
    return decorator


def _ensure_adapter_loaded(name: str):
    """
    确保指定的适配器已加载（延迟加载策略）
    
    通过动态导入模块来触发 @register_adapter 装饰器的执行。
    这样可以保持 __init__.py 为空，符合项目规范。
    
    Args:
        name: 适配器名称
        
    Raises:
        ValueError: 如果适配器不存在
        RuntimeError: 如果模块加载后仍未注册
    """
    if name in _ADAPTER_REGISTRY:
        return  # 已加载
    
    if name not in _ADAPTER_MODULES:
        raise ValueError(
            f"Unknown adapter: {name}. "
            f"Available adapters: {list(_ADAPTER_MODULES.keys())}"
        )
    
    # 动态导入模块，触发 @register_adapter 装饰器执行
    module_path = _ADAPTER_MODULES[name]
    importlib.import_module(module_path)
    
    # 验证注册是否成功
    if name not in _ADAPTER_REGISTRY:
        raise RuntimeError(
            f"Adapter '{name}' module loaded but not registered. "
            f"Check if @register_adapter('{name}') decorator is present."
        )


def create_adapter(name: str, config: dict, output_dir = None) -> BaseAdapter:
    """
    创建适配器实例
    
    Args:
        name: 适配器名称
        config: 配置字典
        output_dir: 输出目录（用于持久化，可选）
        
    Returns:
        适配器实例
        
    Raises:
        ValueError: 如果适配器未注册
    """
    # 延迟加载：确保适配器已加载
    _ensure_adapter_loaded(name)
    
    # 尝试传递 output_dir，如果适配器不支持则回退
    try:
        return _ADAPTER_REGISTRY[name](config, output_dir=output_dir)
    except TypeError:
        # 适配器不接受 output_dir 参数，使用默认方式创建
        return _ADAPTER_REGISTRY[name](config)


def list_adapters() -> List[str]:
    """列出所有可用的适配器"""
    return list(_ADAPTER_MODULES.keys())
