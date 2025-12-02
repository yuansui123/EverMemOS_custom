"""
LongJob 运行器 - 用于启动和管理长任务

提供了运行单个长任务的功能，包括：
- 通过 DI 查找指定的长任务
- 优雅启动和关闭
- 基于 asyncio task cancel 机制处理关闭
- 错误处理和日志记录
"""

import asyncio
from typing import Optional

from core.di.utils import get_bean
from core.longjob.interfaces import LongJobInterface
from core.observation.logger import get_logger

logger = get_logger(__name__)


async def run_longjob_mode(longjob_name: str):
    """
    运行指定的长任务模式

    该函数会作为 asyncio Task 运行，通过 task.cancel() 来触发关闭。
    当收到 CancelledError 时，会优雅地关闭长任务。

    Args:
        longjob_name: 长任务名称
    """
    logger.info("🚀 启动 LongJob 模式: %s", longjob_name)

    longjob_instance: Optional[LongJobInterface] = None

    try:
        # 尝试从 DI 容器中获取指定的长任务
        try:
            longjob_instance = get_bean(longjob_name)
            logger.info(
                "✅ 找到长任务: %s (%s)", longjob_name, type(longjob_instance).__name__
            )
        except Exception as e:
            logger.error("❌ 无法找到长任务 '%s': %s", longjob_name, str(e))
            logger.info("💡 请确保长任务已正确注册到 DI 容器中")
            return

        # 检查是否是 LongJobInterface 的实现
        if not isinstance(longjob_instance, LongJobInterface):
            logger.error("❌ '%s' 不是 LongJobInterface 的实现", longjob_name)
            logger.info("💡 长任务必须继承 LongJobInterface 或其子类")
            return

        # 启动长任务
        logger.info("🔄 启动长任务: %s", longjob_name)
        await longjob_instance.start()

        logger.info("✅ 长任务 '%s' 已启动，正在运行...", longjob_name)

        # 无限等待，直到 task 被 cancel
        # 使用一个永不完成的 Future 来保持任务运行
        await asyncio.Event().wait()

    except asyncio.CancelledError:
        # 收到 task cancel 信号，开始优雅关闭
        logger.info("🛑 收到取消信号，开始优雅关闭长任务: %s", longjob_name)
        if longjob_instance:
            try:
                await longjob_instance.shutdown()
                logger.info("✅ 长任务 '%s' 已成功关闭", longjob_name)
            except Exception as e:
                logger.error("❌ 关闭长任务时出错: %s", str(e), exc_info=True)
        # 重新抛出 CancelledError，让调用方知道任务已被取消
        raise

    except Exception as e:
        # 运行过程中发生异常
        logger.error("❌ 运行长任务时出错: %s", str(e), exc_info=True)
        if longjob_instance:
            try:
                await longjob_instance.shutdown()
                logger.info("✅ 长任务已在异常后关闭")
            except Exception as shutdown_error:
                logger.error(
                    "❌ 关闭长任务时出错: %s", str(shutdown_error), exc_info=True
                )
