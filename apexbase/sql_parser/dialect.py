from enum import Enum

class SQLDialect(Enum):
    SQLITE = "sqlite"
    DUCKDB = "duckdb"
    