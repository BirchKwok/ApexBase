from typing import Literal


def create_storage(
    backend: Literal["sqlite", "duckdb"], 
    filepath: str = None,
    batch_size: int = 1000, 
    enable_cache: bool = True, 
    cache_size: int = 10000
):
    """
    Factory function to create a storage backend instance.
    
    Parameters:
        backend: str
            The storage backend to use ("sqlite" or "duckdb")
        filepath: str
            The file path for storage
        batch_size: int
            The size of batch operations
        enable_cache: bool
            Whether to enable caching
        cache_size: int
            The size of the cache
    """
    if backend == "sqlite":
        from .sqlite_storage import SQLiteStorage
        return SQLiteStorage(filepath, batch_size, enable_cache, cache_size)
    elif backend == "duckdb":
        from .duckdb_storage import DuckDBStorage
        return DuckDBStorage(filepath, batch_size, enable_cache, cache_size)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    