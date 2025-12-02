"""
LongJob 生命周期提供者实现

用于管理长任务的生命周期，包括启动和关闭。
"""

import asyncio
from fastapi import FastAPI
from typing import Optional, Any
import os
from core.observation.logger import get_logger
from core.di.decorators import component
from core.lifespan.lifespan_interface import LifespanProvider

logger = get_logger(__name__)


@component(name="longjob_lifespan_provider")
class LongJobLifespanProvider(LifespanProvider):
    """LongJob 生命周期提供者"""

    def __init__(self, name: str = "longjob", order: int = 100):
        """
        初始化 LongJob 生命周期提供者

        Args:
            name (str): 提供者名称
            order (int): 执行顺序，LongJob 应该在所有基础设施启动后再启动
        """
        super().__init__(name, order)
        self._longjob_task: Optional[asyncio.Task] = None
        self._longjob_name: Optional[str] = None

    async def startup(self, app: FastAPI) -> Any:
        """
        启动 LongJob 任务

        核心逻辑: asyncio.create_task(run_longjob_mode(longjob_name))

        Args:
            app (FastAPI): FastAPI应用实例

        Returns:
            Any: LongJob task 引用
        """
        try:
            from core.longjob.longjob_runner import run_longjob_mode

            self._longjob_name = os.getenv("LONGJOB_NAME")
            if not self._longjob_name:
                logger.warning("⚠️ 未设置 LONGJOB_NAME 环境变量，跳过 LongJob 启动")
                return None
            # 核心逻辑：创建异步任务来运行长任务
            self._longjob_task = asyncio.create_task(
                run_longjob_mode(self._longjob_name)
            )

            # 将 task 存储到 app.state 中，供其他地方访问
            app.state.longjob_task = self._longjob_task

            logger.info("✅ LongJob 任务已启动: %s", self._longjob_name)

            return self._longjob_task

        except Exception as e:
            logger.error("❌ 启动 LongJob 时出错: %s", str(e))
            raise

    async def shutdown(self, app: FastAPI) -> None:
        """
        关闭 LongJob 任务

        Args:
            app (FastAPI): FastAPI应用实例
        """
        if not self._longjob_task:
            logger.info("没有运行中的 LongJob 任务")
            return

        logger.info("正在关闭 LongJob: %s", self._longjob_name)

        try:
            # 取消任务
            if not self._longjob_task.done():
                self._longjob_task.cancel()
                try:
                    await self._longjob_task
                except asyncio.CancelledError:
                    logger.info("✅ LongJob 任务已取消: %s", self._longjob_name)
            else:
                logger.info("✅ LongJob 任务已完成: %s", self._longjob_name)

        except Exception as e:
            logger.error("❌ 关闭 LongJob 时出错: %s", str(e))

        # 清理 app.state 中的 LongJob 相关属性
        if hasattr(app.state, 'longjob_task'):
            delattr(app.state, 'longjob_task')
