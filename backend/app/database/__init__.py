"""
AURA Database Package.

Provides database connection management, transactional helpers, and schema migration utilities.
"""

from .connection import get_db_connection, get_db_context
from .schema import SCHEMA_DDL, init_db

__all__ = [
    "SCHEMA_DDL",
    "get_db_connection",
    "get_db_context",
    "init_db",
]
