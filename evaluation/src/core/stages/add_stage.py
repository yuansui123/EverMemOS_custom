"""
Add 阶段

负责摄入对话数据并构建索引。
"""
from pathlib import Path
from typing import List, Any, Optional
from logging import Logger

from evaluation.src.core.data_models import Conversation, Dataset
from evaluation.src.adapters.base import BaseAdapter
from evaluation.src.utils.checkpoint import CheckpointManager


async def run_add_stage(
    adapter: BaseAdapter,
    dataset: Dataset,
    output_dir: Path,
    checkpoint_manager: Optional[CheckpointManager],
    logger: Logger,
    console: Any,
    completed_stages: set,
) -> dict:
    """
    执行 Add 阶段
    
    Args:
        adapter: 系统适配器
        dataset: 标准格式数据集
        output_dir: 输出目录
        checkpoint_manager: 断点续传管理器
        logger: 日志器
        console: 控制台对象
        completed_stages: 已完成的阶段集合
        
    Returns:
        包含 index 的字典
    """
    # 传递 checkpoint_manager 以支持细粒度断点续传
    index = await adapter.add(
        conversations=dataset.conversations,
        output_dir=output_dir,
        checkpoint_manager=checkpoint_manager
    )
    
    # 索引元数据（延迟加载，无需持久化）
    logger.info("✅ Stage 1 completed")
    
    # 保存 checkpoint
    completed_stages.add("add")
    if checkpoint_manager:
        checkpoint_manager.save_checkpoint(completed_stages)
    
    return {"index": index}

