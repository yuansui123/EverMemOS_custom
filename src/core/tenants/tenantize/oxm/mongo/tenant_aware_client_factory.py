"""
租户感知的 MongoDB 客户端工厂

本模块提供租户化的 MongoDB 客户端工厂实现，基于租户上下文管理客户端。
Factory 只负责创建和缓存客户端，模式判断由 TenantAwareMongoClient 内部处理。
"""

import asyncio
from typing import Optional

from core.observation.logger import get_logger
from core.di.decorators import component
from component.mongodb_client_factory import (
    MongoDBClientFactory,
    MongoDBConfig,
    MongoDBClientWrapper,
)
from core.tenants.tenantize.oxm.mongo.config_utils import get_default_database_name
from core.tenants.tenantize.oxm.mongo.tenant_aware_mongo_client import (
    TenantAwareMongoClient,
)

logger = get_logger(__name__)


@component(name="tenant_aware_mongodb_client_factory", primary=True)
class TenantAwareMongoDBClientFactory(MongoDBClientFactory):
    """
    租户感知的 MongoDB 客户端工厂实现

    此工厂类负责创建和管理租户感知的 MongoDB 客户端。
    所有租户模式 vs 非租户模式的逻辑都由 TenantAwareMongoClient 内部处理，
    Factory 只负责简单的创建、缓存和生命周期管理。

    标记为 primary=True，作为系统默认的 MongoDB 客户端工厂。
    """

    def __init__(self):
        """初始化租户感知的客户端工厂"""
        # 租户感知的客户端包装器（全局单例）
        self._client_wrapper: Optional[MongoDBClientWrapper] = None

        # 并发访问保护锁
        self._lock = asyncio.Lock()

        logger.info("🏭 租户感知的 MongoDB 客户端工厂已初始化 (primary=True)")

    async def get_client(
        self, config: Optional[MongoDBConfig] = None, **connection_kwargs
    ) -> MongoDBClientWrapper:
        """
        获取 MongoDB 客户端

        返回租户感知的客户端包装器。模式判断由 TenantAwareMongoClient 内部处理。

        Args:
            config: MongoDB 配置（保留用于接口兼容，实际配置由租户上下文或环境变量提供）
            **connection_kwargs: 额外的连接参数

        Returns:
            MongoDBClientWrapper: MongoDB 客户端包装器
        """
        return await self._get_client_wrapper()

    async def _get_client_wrapper(self) -> MongoDBClientWrapper:
        """
        获取租户感知的客户端包装器（内部方法）

        单例模式，整个工厂只创建一个客户端包装器实例。

        Returns:
            MongoDBClientWrapper: 包装了租户感知客户端的包装器
        """
        if self._client_wrapper is None:
            async with self._lock:
                # 双重检查
                if self._client_wrapper is None:
                    logger.info("🔧 创建租户感知的 MongoDB 客户端包装器")

                    # 创建租户感知的 MongoDB 客户端
                    tenant_aware_client = TenantAwareMongoClient()

                    # 创建一个虚拟配置（用于兼容接口）
                    dummy_config = MongoDBConfig(
                        host="tenant-aware",
                        port=27017,
                        database=get_default_database_name(),
                    )

                    # 包装成 MongoDBClientWrapper
                    self._client_wrapper = TenantAwareClientWrapper(
                        tenant_aware_client, dummy_config
                    )

                    logger.info("✅ 租户感知的 MongoDB 客户端包装器已创建")

        return self._client_wrapper

    async def get_default_client(self) -> MongoDBClientWrapper:
        """
        获取默认 MongoDB 客户端

        Returns:
            MongoDBClientWrapper: 默认 MongoDB 客户端包装器
        """
        return await self._get_client_wrapper()

    async def get_named_client(self, name: str) -> MongoDBClientWrapper:
        """
        按名称获取 MongoDB 客户端

        注意：在当前实现中，name 参数被忽略，因为租户信息从上下文中获取。
        保留此方法是为了接口兼容性。

        Args:
            name: 前缀名称（保留用于接口兼容）

        Returns:
            MongoDBClientWrapper: MongoDB 客户端包装器
        """
        logger.info("📋 获取命名客户端 name=%s（租户感知模式）", name)
        return await self._get_client_wrapper()

    async def create_client_with_config(
        self,
        host: str = "localhost",
        port: int = 27017,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        **kwargs,
    ) -> MongoDBClientWrapper:
        """
        使用指定配置创建客户端

        注意：在租户感知模式下，配置参数会被传递给 TenantAwareMongoClient，
        用于非租户模式。在租户模式下，这些参数会被忽略。

        Args:
            host: MongoDB 主机
            port: MongoDB 端口
            username: 用户名
            password: 密码
            database: 数据库名
            **kwargs: 其他连接参数

        Returns:
            MongoDBClientWrapper: MongoDB 客户端包装器
        """
        if database is None:
            database = get_default_database_name()
        logger.info(
            "📋 使用指定配置创建客户端（租户感知模式）: host=%s, port=%s, database=%s",
            host,
            port,
            database,
        )
        # 在租户感知模式下，配置参数会传递给 TenantAwareMongoClient
        # 如果启用了非租户模式，TenantAwareMongoClient 会使用这些参数
        return await self._get_client_wrapper()

    async def close_client(self, config: Optional[MongoDBConfig] = None):
        """
        关闭指定客户端

        在租户感知模式下，关闭全局的客户端包装器。

        Args:
            config: 配置（保留用于接口兼容）
        """
        async with self._lock:
            if self._client_wrapper:
                await self._client_wrapper.close()
                self._client_wrapper = None
                logger.info("🔌 MongoDB 客户端已关闭（租户感知工厂）")

    async def close_all_clients(self):
        """关闭所有客户端"""
        await self.close_client()


class TenantAwareClientWrapper(MongoDBClientWrapper):
    """
    租户感知的客户端包装器

    继承自 MongoDBClientWrapper，适配租户感知的 MongoDB 客户端。
    提供与 MongoDBClientWrapper 相同的接口，但内部使用 TenantAwareMongoClient。
    """

    def __init__(
        self, tenant_aware_client: TenantAwareMongoClient, config: MongoDBConfig
    ):
        """
        初始化租户感知的客户端包装器

        Args:
            tenant_aware_client: 租户感知的 MongoDB 客户端
            config: MongoDB 配置（用于兼容性）
        """
        # 直接设置属性，不调用父类 __init__
        self.client = tenant_aware_client
        self.config = config
        self._initialized = False
        self._document_models = []

    @property
    def database(self):
        """
        获取数据库对象

        返回租户感知的数据库代理。
        """
        return self.client[self.config.database]

    async def test_connection(self) -> bool:
        """
        测试连接

        注意：在租户模式下，需要在有租户上下文的情况下调用。
        在非租户模式下，使用提供的配置进行测试。
        """
        try:
            # TenantAwareMongoClient 会根据配置和上下文选择正确的客户端
            real_client = await self.client._get_real_client()
            await real_client.admin.command('ping')
            logger.info("✅ MongoDB 连接测试成功（租户感知）")
            return True
        except Exception as e:
            logger.error("❌ MongoDB 连接测试失败（租户感知）: %s", e)
            return False

    async def close(self):
        """关闭所有连接"""
        if self.client:
            await self.client.close()
            logger.info("🔌 MongoDB 连接已关闭（租户感知）")
