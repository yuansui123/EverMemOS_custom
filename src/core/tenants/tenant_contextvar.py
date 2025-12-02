"""
租户上下文管理模块

本模块提供了基于 contextvars 的租户上下文管理功能，
用于在异步环境中安全地存储和访问当前租户信息。
"""

from contextvars import ContextVar
from typing import Optional

from src.core.tenants.tenant_models import TenantInfo


# 全局的租户上下文变量
# 使用 ContextVar 确保在异步环境中每个任务都有独立的租户上下文
current_tenant_contextvar: ContextVar[Optional[TenantInfo]] = ContextVar(
    'current_tenant', default=None
)


def set_current_tenant(tenant_info: Optional[TenantInfo]) -> None:
    """
    设置当前请求的租户信息

    此方法将租户信息设置到当前上下文中。在异步环境下，
    每个请求/任务都有独立的上下文，互不干扰。

    Args:
        tenant_info: 租户信息对象，如果为 None 则清除当前租户信息

    Examples:
        >>> from src.core.tenants.tenant_models import TenantInfo, TenantDetail
        >>> tenant = TenantInfo(
        ...     tenant_id="tenant_001",
        ...     tenant_detail=TenantDetail(),
        ...     origin_tenant_data={}
        ... )
        >>> set_current_tenant(tenant)
    """
    current_tenant_contextvar.set(tenant_info)


def get_current_tenant() -> Optional[TenantInfo]:
    """
    获取当前请求的租户信息

    此方法从当前上下文中获取租户信息。如果当前上下文中没有设置租户信息，
    则返回 None。

    Returns:
        当前上下文中的租户信息，如果未设置则返回 None

    Examples:
        >>> tenant = get_current_tenant()
        >>> if tenant:
        ...     print(f"当前租户ID: {tenant.tenant_id}")
        ... else:
        ...     print("未设置租户信息")
    """
    return current_tenant_contextvar.get()


def clear_current_tenant() -> None:
    """
    清除当前请求的租户信息

    此方法将当前上下文中的租户信息设置为 None，
    相当于 set_current_tenant(None)。

    Examples:
        >>> clear_current_tenant()
    """
    current_tenant_contextvar.set(None)


def get_current_tenant_id() -> Optional[str]:
    """
    获取当前租户的 ID

    这是一个便捷方法，直接返回当前租户的 tenant_id。
    如果当前没有设置租户信息，则返回 None。

    Returns:
        当前租户的 ID，如果未设置租户信息则返回 None

    Examples:
        >>> tenant_id = get_current_tenant_id()
        >>> if tenant_id:
        ...     print(f"当前租户ID: {tenant_id}")
    """
    tenant = get_current_tenant()
    return tenant.tenant_id if tenant else None
