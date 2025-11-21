"""MemCell 同步服务

负责将 MemCell 同步到 Milvus 和 Elasticsearch（群组记忆）。
语义记忆与事件日志由 MemorySyncService 处理。
"""
from typing import Dict, Optional, List, Any
from core.di import get_bean_by_type, service
from core.observation.logger import get_logger
from infra_layer.adapters.out.persistence.document.memory.memcell import MemCell
from infra_layer.adapters.out.search.repository.episodic_memory_milvus_repository import (
    EpisodicMemoryMilvusRepository,
)
from infra_layer.adapters.out.search.repository.episodic_memory_es_repository import (
    EpisodicMemoryEsRepository,
)
from infra_layer.adapters.out.search.repository.semantic_memory_milvus_repository import (
    SemanticMemoryMilvusRepository,
)
from infra_layer.adapters.out.search.repository.semantic_memory_es_repository import (
    SemanticMemoryEsRepository,
)
from infra_layer.adapters.out.search.repository.event_log_milvus_repository import (
    EventLogMilvusRepository,
)
from infra_layer.adapters.out.search.repository.event_log_es_repository import (
    EventLogEsRepository,
)
from common_utils.datetime_utils import get_now_with_timezone

logger = get_logger(__name__)


@service(name="memcell_sync_service", primary=True)
class MemCellSyncService:
    """MemCell 同步服务
    
    负责将 MemCell 中的各项数据同步到 Milvus 和 Elasticsearch：
    1. episode -> EpisodicMemory
    2. semantic_memories -> SemanticMemory
    3. event_log -> EventLog
    """

    def __init__(
        self,
    ):
        """初始化同步服务
        
        Args:
            episodic_milvus_repo: 情景记忆 Milvus 仓库实例（可选，不提供则从 DI 获取）
            es_repo: ES 仓库实例（可选，不提供则从 DI 获取）
        """
        # self.episodic_milvus_repo = episodic_milvus_repo or get_bean_by_type(
        #     EpisodicMemoryMilvusRepository
        # )
        # self.es_repo = es_repo or get_bean_by_type(EpisodicMemoryEsRepository)
        self._episodic_es_repo = get_bean_by_type(EpisodicMemoryEsRepository)
        self._episodic_milvus_repo = get_bean_by_type(EpisodicMemoryMilvusRepository)
        self._semantic_milvus_repo = get_bean_by_type(SemanticMemoryMilvusRepository)
        self._semantic_es_repo = get_bean_by_type(SemanticMemoryEsRepository)
        self._eventlog_milvus_repo = get_bean_by_type(EventLogMilvusRepository)
        self._eventlog_es_repo = get_bean_by_type(EventLogEsRepository)
        logger.info("MemCellSyncService 初始化完成")

    async def sync_memcell(
        self, memcell: MemCell, sync_to_es: bool = True, sync_to_milvus: bool = True
    ) -> Dict[str, int]:
        """同步单个 MemCell 的所有内容到 Milvus 和 ES"""
        stats = {"episode": 0, "semantic_memory": 0, "event_log": 0, "es_records": 0}
        
        try:
            # 1. 同步 Episode
            if hasattr(memcell, 'episode') and memcell.episode:
                episode_stats = await self._sync_episode(memcell, sync_to_es, sync_to_milvus)
                stats["episode"] += episode_stats["count"]
                stats["es_records"] += episode_stats["es_count"]
            
            # 2. 同步 Semantic Memories
            if hasattr(memcell, 'semantic_memories') and memcell.semantic_memories:
                sem_stats = await self._sync_semantic_memories(
                    memcell, sync_to_es, sync_to_milvus
                )
                stats["semantic_memory"] += sem_stats["count"]
                stats["es_records"] += sem_stats["es_count"]

            # 3. 同步 Event Log
            if hasattr(memcell, 'event_log') and memcell.event_log:
                log_stats = await self._sync_event_log(
                    memcell, sync_to_es, sync_to_milvus
                )
                stats["event_log"] += log_stats["count"]
                stats["es_records"] += log_stats["es_count"]

            # 刷新操作
            if sync_to_milvus:
                await self._episodic_milvus_repo.flush()
                if stats["semantic_memory"] > 0:
                    await self._semantic_milvus_repo.flush()
                if stats["event_log"] > 0:
                    await self._eventlog_milvus_repo.flush()
            
            # ES 刷新
            if sync_to_es and stats["es_records"] > 0:
                try:
                    client = await self._episodic_es_repo.get_client()
                    # 刷新所有相关索引
                    await client.indices.refresh(index="_all")
                except Exception:
                    pass

            logger.info(f"MemCell 同步完成: {memcell.event_id}, 统计: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"MemCell 同步失败: {memcell.event_id}, error={e}", exc_info=True)
            raise

    async def _sync_episode(self, memcell: MemCell, sync_to_es: bool = True, sync_to_milvus: bool = True) -> None:
        """同步 episode 到 Milvus 和 ES"""
        stats = {"count": 0, "es_count": 0}
        if sync_to_milvus:
            # 从 MongoDB 读取 embedding
            vector = None
            if hasattr(memcell, 'extend') and memcell.extend and 'embedding' in memcell.extend:
                vector = memcell.extend['embedding']
                if hasattr(vector, 'tolist'):
                    vector = vector.tolist()
            
            if not vector:
                logger.warning(f"episode 缺少 embedding，跳过: {memcell.event_id}")
            else:    
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
                await self._episodic_milvus_repo.create_and_save_episodic_memory(
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
            
        if sync_to_es:
            """同步 episode 到 ES"""
            search_content = []
            if hasattr(memcell, 'subject') and memcell.subject:
                search_content.append(memcell.subject)
            if hasattr(memcell, 'summary') and memcell.summary:
                search_content.append(memcell.summary)
            search_content.append(memcell.episode[:500])
            
            await self._episodic_es_repo.create_and_save_episodic_memory(
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
                event_type="episode",
                subject=getattr(memcell, 'subject', None),
                memcell_event_id_list=[str(memcell.event_id)],
            )
            logger.debug(f"✅ 已同步 episode 到 ES: {memcell.event_id}")
            stats["es_count"] += 1
        stats["count"] += 1
        return stats
    

    async def _sync_semantic_memories(
        self, memcell: MemCell, sync_to_es: bool, sync_to_milvus: bool
    ) -> Dict[str, int]:
        """同步 MemCell 中的语义记忆"""
        stats = {"count": 0, "es_count": 0}
        
        for idx, sem_mem in enumerate(memcell.semantic_memories):
            try:
                content = sem_mem.get("content") if isinstance(sem_mem, dict) else getattr(sem_mem, "content", None)
                embedding = sem_mem.get("embedding") if isinstance(sem_mem, dict) else getattr(sem_mem, "embedding", None)
                user_id = sem_mem.get("user_id") if isinstance(sem_mem, dict) else getattr(sem_mem, "user_id", memcell.user_id)
                
                if not content or not embedding:
                    continue

                sem_id = f"{memcell.event_id}_sem_{idx}"
                # 确保 embedding 是 list
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()
                
                parent_episode_id = str(memcell.event_id)

                if sync_to_milvus:
                    await self._semantic_milvus_repo.create_and_save_semantic_memory(
                        event_id=sem_id,
                        user_id=user_id or "",
                        content=content,
                        parent_episode_id=parent_episode_id,
                        vector=embedding,
                        group_id=memcell.group_id,
                        start_time=memcell.timestamp,
                        end_time=memcell.timestamp,
                    )
                
                if sync_to_es:
                    await self._semantic_es_repo.create_and_save_semantic_memory(
                        event_id=sem_id,
                        user_id=str(user_id) if user_id else "unknown",
                        timestamp=memcell.timestamp or get_now_with_timezone(),
                        content=content,
                        search_content=[content],
                        parent_episode_id=parent_episode_id,
                        group_id=memcell.group_id,
                        start_time=memcell.timestamp,
                        end_time=memcell.timestamp,
                    )
                    stats["es_count"] += 1
                
                stats["count"] += 1
                
            except Exception as e:
                logger.error(f"同步语义记忆失败: {e}", exc_info=True)
                
        return stats

    async def _sync_event_log(
        self, memcell: MemCell, sync_to_es: bool, sync_to_milvus: bool
    ) -> Dict[str, int]:
        """同步 MemCell 中的 Event Log"""
        stats = {"count": 0, "es_count": 0}
        
        try:
            event_logs = memcell.event_log
            if not event_logs:
                return stats
                

            for idx, atomic_fact in enumerate(event_logs.get("atomic_fact")):
                embedding = event_logs.get("fact_embeddings")[idx]
                user_id = memcell.user_id

                if not atomic_fact and not embedding:
                    continue

                event_id = f"{memcell.event_id}_evt_{idx}"
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()
                
                if sync_to_milvus:
                    await self._eventlog_milvus_repo.create_and_save_event_log(
                        event_id=event_id,
                        user_id = user_id or "",
                        atomic_fact=atomic_fact,
                        parent_episode_id=str(memcell.event_id),
                        timestamp=memcell.timestamp or get_now_with_timezone(),
                        vector=embedding,
                        group_id=memcell.group_id
                    )
                
                if sync_to_es:
                    await self._eventlog_es_repo.create_and_save_event_log(
                        event_id=event_id,
                        user_id=user_id or "",
                        user_name=getattr(memcell, "user_name", None) or "",
                        group_name=getattr(memcell, "group_name", None) or "",
                        atomic_fact=atomic_fact,
                        search_content=[atomic_fact],
                        parent_episode_id=str(memcell.event_id),
                        timestamp=memcell.timestamp or get_now_with_timezone(),
                        group_id=memcell.group_id
                    )
                    stats["es_count"] += 1
                
                stats["count"] += 1
            
        except Exception as e:
            logger.error(f"同步 Event Log 失败: {e}", exc_info=True)
            
        return stats

    async def sync_memcells_batch(self, memcells: List[MemCell]) -> Dict[str, Any]:
        """批量同步 MemCells 到 Milvus"""
        total_stats = {
            "total_memcells": len(memcells),
            "success_memcells": 0,
            "failed_memcells": 0,
            "total_episode": 0,
            "total_semantic": 0,
            "total_event_log": 0
        }
        
        for memcell in memcells:
            try:
                stats = await self.sync_memcell(memcell)
                total_stats["success_memcells"] += 1
                total_stats["total_episode"] += stats.get("episode", 0)
                total_stats["total_semantic"] += stats.get("semantic_memory", 0)
                total_stats["total_event_log"] += stats.get("event_log", 0)
            except Exception as e:
                logger.error(f"批量同步失败: {memcell.event_id}, error={e}", exc_info=True)
                total_stats["failed_memcells"] += 1
                continue
        
        logger.info(f"批量同步完成: {total_stats}")
        return total_stats


def get_memcell_sync_service() -> MemCellSyncService:
    """获取 MemCell 同步服务实例
    
    通过依赖注入框架获取服务实例，支持单例模式。
    """
    from core.di import get_bean
    return get_bean("memcell_sync_service")
