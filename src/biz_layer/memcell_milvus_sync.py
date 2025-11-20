"""MemCell 到 Milvus 同步服务

负责将 MemCell.episode 同步到 Milvus 向量数据库（群组记忆）。
PersonalSemanticMemory 和 PersonalEventLog 由 PersonalMemorySyncService 处理。
"""

from typing import Optional, List, Dict, Any
import logging
from datetime import datetime

from infra_layer.adapters.out.persistence.document.memory.memcell import MemCell
from infra_layer.adapters.out.search.repository.episodic_memory_milvus_repository import (
    EpisodicMemoryMilvusRepository,
)
from infra_layer.adapters.out.search.repository.episodic_memory_es_repository import (
    EpisodicMemoryEsRepository,
)
from agentic_layer.vectorize_service import DeepInfraVectorizeServiceInterface
from core.di import get_bean_by_type, service
from common_utils.datetime_utils import get_now_with_timezone

logger = logging.getLogger(__name__)


@service(name="memcell_milvus_sync_service", primary=True)
class MemCellMilvusSyncService:
    """MemCell 到 Milvus 同步服务

    只负责将 MemCell.episode 同步到 Milvus（群组记忆）。
    PersonalSemanticMemory 和 PersonalEventLog 由 PersonalMemorySyncService 处理。
    """

    def __init__(
        self,
        episodic_milvus_repo: Optional[EpisodicMemoryMilvusRepository] = None,
        es_repo: Optional[EpisodicMemoryEsRepository] = None,
        vectorize_service: Optional[DeepInfraVectorizeServiceInterface] = None,
    ):
        """初始化同步服务

        Args:
            episodic_milvus_repo: 情景记忆 Milvus 仓库实例（可选，不提供则从 DI 获取）
            es_repo: ES 仓库实例（可选，不提供则从 DI 获取）
            vectorize_service: 向量化服务实例（可选，不提供则从 DI 获取）
        """
        self.episodic_milvus_repo = episodic_milvus_repo or get_bean_by_type(
            EpisodicMemoryMilvusRepository
        )
        self.es_repo = es_repo or get_bean_by_type(EpisodicMemoryEsRepository)

        if vectorize_service is None:
            from agentic_layer.vectorize_service import get_vectorize_service

            self.vectorize_service = get_vectorize_service()
        else:
            self.vectorize_service = vectorize_service

        logger.info(
            "MemCellMilvusSyncService 初始化完成（仅同步 episode 到 Milvus + ES）"
        )

    async def sync_memcell(
        self, memcell: MemCell, sync_to_es: bool = True, sync_to_milvus: bool = True
    ) -> Dict[str, int]:
        """同步单个 MemCell.episode 到 Milvus 和 ES

        Args:
            memcell: MemCell 文档对象
            sync_to_es: 是否同步到 ES（默认 True）
            sync_to_milvus: 是否同步到 Milvus（默认 True）

        Returns:
            同步统计信息 {"episode": 1}
        """
        stats = {"episode": 0, "es_records": 0}

        try:
            # 同步到 Milvus
            if sync_to_milvus:
                # 只同步 episode（群组记忆）
                if hasattr(memcell, 'episode') and memcell.episode:
                    await self._sync_episode(memcell)
                    stats["episode"] = 1
                    logger.debug(f"✅ 已同步 episode 到 Milvus: {memcell.event_id}")

            # 同步到 ES
            if sync_to_es:
                es_count = await self._sync_to_es(memcell)
                stats["es_records"] = es_count
                logger.debug(f"✅ 已同步 {es_count} 条记录到 ES: {memcell.event_id}")

            logger.info(f"MemCell 同步完成: {memcell.event_id}, 统计: {stats}")

            return stats

        except Exception as e:
            logger.error(f"MemCell 同步失败: {memcell.event_id}, error={e}")
            raise

    async def _sync_episode(self, memcell: MemCell) -> None:
        """同步 episode 到 Milvus

        Args:
            memcell: MemCell 文档对象
        """
        # 从 MongoDB 读取 embedding（必须存在）
        vector = None
        if (
            hasattr(memcell, 'extend')
            and memcell.extend
            and 'embedding' in memcell.extend
        ):
            vector = memcell.extend['embedding']
            # 确保是 list 格式（可能是 numpy array）
            if hasattr(vector, 'tolist'):
                vector = vector.tolist()
            logger.debug(f"从 MongoDB 读取 episode embedding: {memcell.event_id}")

        if not vector:
            logger.warning(
                f"episode 缺少 embedding，跳过同步到 Milvus: {memcell.event_id}"
            )
            return

        # 准备搜索内容
        search_content = []
        if hasattr(memcell, 'subject') and memcell.subject:
            search_content.append(memcell.subject)
        if hasattr(memcell, 'summary') and memcell.summary:
            search_content.append(memcell.summary)
        if not search_content:
            search_content.append(memcell.episode[:100])  # 使用 episode 前 100 字符

        # 确保 vector 是 list 格式
        if hasattr(vector, 'tolist'):
            vector = vector.tolist()

        # MemCell 的 user_id 始终为 None（群组记忆）
        await self.episodic_milvus_repo.create_and_save_episodic_memory(
            event_id=str(memcell.event_id),
            user_id=memcell.user_id,  # None for group memory
            timestamp=memcell.timestamp or get_now_with_timezone(),
            episode=memcell.episode,
            search_content=search_content,
            vector=vector,
            user_name=memcell.user_id,
            title=getattr(memcell, 'subject', None),
            summary=getattr(memcell, 'summary', None),
            group_id=getattr(memcell, 'group_id', None),
            participants=getattr(memcell, 'participants', None),
            subject=getattr(memcell, 'subject', None),
            metadata="{}",
            memcell_event_id_list=[str(memcell.event_id)],
        )
        logger.debug(f"✅ 已同步 episode 到 Milvus: {memcell.event_id}")

    async def _sync_to_es(self, memcell: MemCell) -> int:
        """同步 MemCell.episode 到 ES

        Args:
            memcell: MemCell 文档对象

        Returns:
            同步的记录数量
        """
        count = 0

        try:
            # 只同步 episode（群组记忆）
            if hasattr(memcell, 'episode') and memcell.episode:
                search_content = []
                if hasattr(memcell, 'subject') and memcell.subject:
                    search_content.append(memcell.subject)
                if hasattr(memcell, 'summary') and memcell.summary:
                    search_content.append(memcell.summary)
                search_content.append(memcell.episode[:500])

                await self.es_repo.create_and_save_episodic_memory(
                    event_id=f"{str(memcell.event_id)}_episode",
                    user_id=memcell.user_id,
                    timestamp=memcell.timestamp or get_now_with_timezone(),
                    episode=memcell.episode,
                    search_content=search_content,
                    user_name=memcell.user_id,
                    title=getattr(memcell, 'subject', None),
                    summary=getattr(memcell, 'summary', None),
                    group_id=getattr(memcell, 'group_id', None),
                    participants=getattr(memcell, 'participants', None),
                    event_type="episode",  # 标记类型
                    subject=getattr(memcell, 'subject', None),
                    memcell_event_id_list=[str(memcell.event_id)],
                )
                count += 1

            return count

        except Exception as e:
            logger.error(f"同步到 ES 失败: {memcell.event_id}, error={e}")
            return 0

    async def sync_memcells_batch(self, memcells: List[MemCell]) -> Dict[str, Any]:
        """批量同步 MemCells 到 Milvus

        Args:
            memcells: MemCell 文档对象列表

        Returns:
            批量同步统计信息
        """
        total_stats = {
            "total_memcells": len(memcells),
            "success_memcells": 0,
            "failed_memcells": 0,
            "total_episode": 0,
        }

        for memcell in memcells:
            try:
                stats = await self.sync_memcell(memcell)
                total_stats["success_memcells"] += 1
                total_stats["total_episode"] += stats["episode"]
            except Exception as e:
                logger.error(f"批量同步失败: {memcell.event_id}, error={e}")
                total_stats["failed_memcells"] += 1
                continue

        logger.info(f"批量同步完成: {total_stats}")
        return total_stats


def get_memcell_milvus_sync_service() -> MemCellMilvusSyncService:
    """获取 MemCell Milvus 同步服务实例

    通过依赖注入框架获取服务实例，支持单例模式。
    """
    from core.di import get_bean

    return get_bean("memcell_milvus_sync_service")
