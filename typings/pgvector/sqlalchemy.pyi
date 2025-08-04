# Type stubs for pgvector.sqlalchemy
from typing import Any, Type
from sqlalchemy.types import UserDefinedType

class VECTOR(UserDefinedType[Any]):
    cache_ok: bool = True

    def __init__(self, dim: int = None) -> None: ...

# Alias for compatibility
Vector: Type[VECTOR] = VECTOR
