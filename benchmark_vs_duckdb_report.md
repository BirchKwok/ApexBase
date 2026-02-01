# ApexBase vs DuckDB Benchmark Report

## Test Environment

- **Platform**: macOS-26.2-arm64-arm-64bit
- **Processor**: arm
- **Machine**: arm64
- **Python Version**: 3.11.10
- **CPU**: Apple M1 Pro
- **Memory**: 32.0 GB
- **ApexBase Version**: 0.4.0
- **DuckDB Version**: 1.1.3

## Performance Comparison: ApexBase vs DuckDB

**Dataset**: 1,000,000 rows
**Iterations**: 5 (after 2 warmup)

| Query | ApexBase | DuckDB | Ratio |
|-------|----------|--------|-------|
| COUNT(*) | 0.08ms | 0.37ms | 0.22x |
| Filter (age > 50) | 383.54ms | 179.55ms | 2.14x |
| Filter (name = 'user_5000') | 7.41ms | 6.35ms | 1.17x |
| GROUP BY age (100 groups) | 3.44ms | 0.98ms | 3.51x |
| GROUP BY name, age (10K groups) | 43.47ms | 8.51ms | 5.11x |
| Insert 1M rows | 327.44ms | 197844.50ms | 0.00x |
| ORDER BY + LIMIT 100 | 7.45ms | 1.19ms | 6.26x |
| ORDER BY + LIMIT 1K | 25.55ms | 5.26ms | 4.86x |
| SELECT * LIMIT 100 | 0.09ms | 0.25ms | 0.35x |
| SELECT * LIMIT 10K | 0.26ms | 3.53ms | 0.07x |

## Notes

- Lower time is better
- Ratio > 1 means ApexBase is slower than DuckDB
- Ratio < 1 means ApexBase is faster than DuckDB
