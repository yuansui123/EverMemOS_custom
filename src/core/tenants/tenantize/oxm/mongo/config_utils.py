"""
MongoDB 配置工具函数

提供租户感知的 MongoDB 配置相关的公共工具函数。
"""

import os
from typing import Optional, Dict, Any

from core.observation.logger import get_logger
from core.tenants.tenant_contextvar import get_current_tenant

logger = get_logger(__name__)


def get_tenant_mongo_config() -> Optional[Dict[str, Any]]:
    """
    获取当前租户的 MongoDB 配置信息

    这是一个公共工具函数，用于从租户上下文中提取 MongoDB 配置。

    Returns:
        Optional[Dict[str, Any]]: MongoDB 配置字典，如果无法获取则返回 None

    配置字典可能包含的字段：
        - uri: MongoDB 连接 URI
        - host: MongoDB 主机地址
        - port: MongoDB 端口
        - username: 用户名
        - password: 密码
        - database: 数据库名称
        - 其他连接参数
    """
    tenant_info = get_current_tenant()
    if not tenant_info:
        logger.debug("⚠️ 无法获取租户信息")
        return None

    mongo_config = tenant_info.get_storage_info("mongodb")
    if not mongo_config:
        logger.debug("⚠️ 租户配置中缺少 MongoDB 信息")
        return None

    return mongo_config


def get_mongo_client_cache_key(config: Dict[str, Any]) -> str:
    """
    根据 MongoDB 配置生成缓存键

    基于连接参数（host/port/username/password/uri）生成唯一的缓存键，
    这样相同配置的连接可以复用同一个客户端实例。

    Args:
        config: MongoDB 配置字典

    Returns:
        str: 缓存键
    """
    # 优先使用 URI 生成缓存键
    uri = config.get("uri")
    if uri:
        # 对于 URI，直接使用其作为主要标识
        # 注意：URI 中可能包含敏感信息，但这只是内存中的缓存键
        return f"uri:{uri}"

    # 使用 host/port/username 组合生成缓存键
    host = config.get("host", "localhost")
    port = config.get("port", 27017)
    username = config.get("username", "")

    # 不包含 password 在缓存键中（因为 password 相同时，其他参数也应该相同）
    # 不包含 database 在缓存键中（同一个客户端可以访问多个数据库）
    cache_key = f"host:{host}:port:{port}:user:{username}"

    return cache_key


def load_mongo_config_from_env() -> Dict[str, Any]:
    """
    从环境变量加载 MongoDB 配置

    读取 MONGODB_* 环境变量，优先使用 MONGODB_URI。
    用于后备客户端或默认客户端的配置加载。

    Returns:
        Dict[str, Any]: 包含连接信息的配置字典

    环境变量：
        - MONGODB_URI: MongoDB 连接 URI（优先使用）
        - MONGODB_HOST: MongoDB 主机地址（默认：localhost）
        - MONGODB_PORT: MongoDB 端口（默认：27017）
        - MONGODB_USERNAME: 用户名（可选）
        - MONGODB_PASSWORD: 密码（可选）
        - MONGODB_DATABASE: 数据库名称（默认：memsys）
    """
    # 优先使用 MONGODB_URI
    uri = os.getenv("MONGODB_URI")
    if uri:
        logger.info("📋 从环境变量 MONGODB_URI 加载配置")
        return {"uri": uri, "database": get_default_database_name()}

    # 分别读取各个配置项
    host = os.getenv("MONGODB_HOST", "localhost")
    port = int(os.getenv("MONGODB_PORT", "27017"))
    username = os.getenv("MONGODB_USERNAME")
    password = os.getenv("MONGODB_PASSWORD")
    database = get_default_database_name()

    logger.info(
        "📋 从环境变量加载配置: host=%s, port=%s, database=%s", host, port, database
    )

    return {
        "host": host,
        "port": port,
        "username": username,
        "password": password,
        "database": database,
    }


def get_default_database_name() -> str:
    """
    获取默认的数据库名称

    从环境变量 MONGODB_DATABASE 读取，如果未设置则返回 "memsys"。

    Returns:
        str: 默认的数据库名称
    """
    return os.getenv("MONGODB_DATABASE", "memsys")
