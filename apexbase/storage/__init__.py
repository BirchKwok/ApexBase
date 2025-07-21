from typing import Literal


def create_storage(
    filepath: str = None,
    batch_size: int = 1000, 
    enable_cache: bool = True, 
    cache_size: int = 10000
):
    """
    Factory function to create a DuckDB storage backend instance.
    
    Parameters:
        filepath: str
            The file path for storage
        batch_size: int
            The size of batch operations
        enable_cache: bool
            Whether to enable caching
        cache_size: int
            The size of the cache
    """
    from .duckdb_storage import DuckDBStorage
    return DuckDBStorage(filepath, batch_size, enable_cache, cache_size)
    