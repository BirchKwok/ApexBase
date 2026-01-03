# ApexBase Test Suite

This directory contains comprehensive pytest test cases for the ApexBase Python API, providing 100% coverage of all public APIs and thorough testing of edge cases, error handling, and performance scenarios.

## Test Suite Structure

### Core Test Files

| Test File | Description | Coverage |
|-----------|-------------|----------|
| `test_init_config.py` | ApexClient initialization and configuration | ✅ Complete |
| `test_table_management.py` | Table operations (create, drop, list, use) | ✅ Complete |
| `test_data_storage.py` | Data storage with all supported formats | ✅ Complete |
| `test_query_operations.py` | Query operations and ResultView functionality | ✅ Complete |
| `test_sql_execute.py` | SQL execute operations and SqlResult | ✅ Complete |
| `test_retrieve_operations.py` | Retrieve operations (single, many, all) | ✅ Complete |
| `test_fts_operations.py` | Full-text search functionality | ✅ Complete |
| `test_data_modification.py` | Data modification (delete, replace) | ✅ Complete |
| `test_column_management.py` | Column management operations | ✅ Complete |
| `test_edge_cases.py` | Edge cases and error handling | ✅ Complete |
| `test_lifecycle_management.py` | Context manager and lifecycle | ✅ Complete |
| `test_data_conversions.py` | Data format conversions | ✅ Complete |

## Running Tests

### Quick Start

```bash
# Run all tests
python run_tests.py

# Run with verbose output
python run_tests.py --verbose

# Run with coverage reporting
python run_tests.py --coverage

# Run tests in parallel
python run_tests.py --parallel
```

### Advanced Usage

```bash
# Run specific test file
python run_tests.py --file test_init_config.py

# Run tests with specific markers
python run_tests.py --markers "fts"

# Skip slow tests
python run_tests.py --skip-slow

# List all available tests
python run_tests.py --list

# Check dependencies
python run_tests.py --check-deps
```

### Direct pytest Usage

```bash
# Run all tests
pytest test/

# Run with coverage
pytest --cov=apexbase --cov-report=html test/

# Run specific test
pytest test/test_init_config.py

# Run with markers
pytest -m "fts" test/

# Run in parallel
pytest -n auto test/
```

## Test Coverage Areas

### 1. ApexClient Initialization (`test_init_config.py`)
- ✅ Default initialization parameters
- ✅ Custom configuration options
- ✅ Durability levels ('fast', 'safe', 'max')
- ✅ Database path handling and creation
- ✅ Error handling for invalid parameters
- ✅ Edge cases (empty paths, large values, etc.)

### 2. Table Management (`test_table_management.py`)
- ✅ Table creation, deletion, listing
- ✅ Table switching and current table tracking
- ✅ Table name validation and special characters
- ✅ Table isolation and data separation
- ✅ FTS integration with table operations
- ✅ Persistence across client sessions

### 3. Data Storage (`test_data_storage.py`)
- ✅ Single record storage (dict)
- ✅ Batch storage (list of dicts)
- ✅ Columnar storage (Dict[str, list])
- ✅ NumPy array storage
- ✅ Pandas DataFrame storage
- ✅ Polars DataFrame storage
- ✅ PyArrow Table storage
- ✅ Performance optimization paths
- ✅ Edge cases and error handling

### 4. Query Operations (`test_query_operations.py`)
- ✅ Basic and complex WHERE clauses
- ✅ Query optimization with limits
- ✅ ResultView functionality and conversions
- ✅ String operations and special characters
- ✅ NULL value handling
- ✅ Performance considerations
- ✅ Arrow optimization paths

### 5. SQL Execute (`test_sql_execute.py`)
- ✅ Complete SQL SELECT statements
- ✅ ORDER BY, LIMIT, DISTINCT, OFFSET
- ✅ Aggregate functions (COUNT, SUM, AVG, MIN, MAX)
- ✅ GROUP BY and HAVING clauses
- ✅ SqlResult functionality and conversions
- ✅ Complex queries and subqueries
- ✅ Performance with large datasets

### 6. Retrieve Operations (`test_retrieve_operations.py`)
- ✅ Single record retrieval
- ✅ Multiple record retrieval
- ✅ All records retrieval
- ✅ Arrow C Data Interface optimization
- ✅ ResultView conversions
- ✅ Performance comparisons
- ✅ Edge cases and error handling

### 7. FTS Operations (`test_fts_operations.py`)
- ✅ FTS initialization and configuration
- ✅ Basic and fuzzy text search
- ✅ Search and retrieve operations
- ✅ Multi-table FTS configurations
- ✅ FTS statistics and management
- ✅ Performance with large datasets
- ✅ Unicode and special character support

### 8. Data Modification (`test_data_modification.py`)
- ✅ Single and batch delete operations
- ✅ Single and batch replace operations
- ✅ Data consistency after modifications
- ✅ FTS index updates
- ✅ Performance considerations
- ✅ Table isolation

### 9. Column Management (`test_column_management.py`)
- ✅ Column addition with various types
- ✅ Column deletion operations
- ✅ Column renaming operations
- ✅ Data type retrieval
- ✅ Edge cases and error handling
- ✅ Large dataset performance

### 10. Edge Cases (`test_edge_cases.py`)
- ✅ Invalid parameter handling
- ✅ Resource exhaustion scenarios
- ✅ Concurrent access patterns
- ✅ Data corruption scenarios
- ✅ Network and I/O error simulation
- ✅ Memory pressure handling
- ✅ Boundary condition testing

### 11. Lifecycle Management (`test_lifecycle_management.py`)
- ✅ Context manager functionality
- ✅ Automatic resource cleanup
- ✅ Exception handling in contexts
- ✅ Instance registry and cleanup
- ✅ Resource leak prevention
- ✅ Concurrent lifecycle operations

### 12. Data Conversions (`test_data_conversions.py`)
- ✅ Pandas DataFrame conversions
- ✅ Polars DataFrame conversions
- ✅ PyArrow Table conversions
- ✅ Cross-format conversions
- ✅ Type preservation
- ✅ Performance with large datasets
- ✅ SqlResult conversions

## Dependencies

### Required Dependencies
- `pytest` - Test framework
- `apexbase` - The library being tested

### Optional Dependencies
- `pytest-cov` - Coverage reporting
- `pytest-xdist` - Parallel test execution
- `pytest-timeout` - Test timeout handling
- `pandas` - Pandas DataFrame tests
- `polars` - Polars DataFrame tests
- `pyarrow` - PyArrow Table tests

### Installing Dependencies

```bash
# Install required dependencies
pip install pytest

# Install optional dependencies for full testing
pip install pytest-cov pytest-xdist pytest-timeout pandas polars pyarrow

# Or install all at once
pip install pytest pytest-cov pytest-xdist pytest-timeout pandas polars pyarrow
```

## Test Configuration

The test suite is configured via `pytest.ini` with the following settings:

- **Test Discovery**: Automatically finds all `test_*.py` files
- **Output Formatting**: Verbose, colored output with short tracebacks
- **Markers**: Predefined markers for test categorization
- **Coverage**: HTML and terminal coverage reports
- **Parallel**: Automatic parallel execution support
- **Timeout**: 5-minute timeout per test
- **Warnings**: Filtered for cleaner output

## Test Markers

Use pytest markers to run specific test categories:

```bash
# Run only FTS tests
pytest -m "fts"

# Run only performance tests
pytest -m "performance"

# Skip slow tests
pytest -m "not slow"

# Run integration tests
pytest -m "integration"

# Run unit tests only
pytest -m "unit"
```

Available markers:
- `slow` - Tests that take longer to run
- `integration` - Integration tests
- `unit` - Unit tests
- `pandas` - Tests requiring pandas
- `polars` - Tests requiring polars
- `pyarrow` - Tests requiring pyarrow
- `fts` - Full-text search tests
- `performance` - Performance measurement tests

## Coverage Reporting

Generate comprehensive coverage reports:

```bash
# Run tests with coverage
python run_tests.py --coverage

# Or directly with pytest
pytest --cov=apexbase --cov-report=html --cov-report=term-missing test/

# View HTML coverage report
open htmlcov/index.html
```

## Performance Testing

The test suite includes performance benchmarks for:

- Large dataset operations (10K+ records)
- Conversion performance between formats
- Concurrent access patterns
- Memory usage optimization
- Arrow C Data Interface efficiency

Run performance tests:

```bash
# Run only performance tests
pytest -m "performance" -v

# Run with timing information
pytest --durations=10 test/
```

## Continuous Integration

The test suite is designed for CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Tests
  run: |
    python run_tests.py --coverage --parallel

- name: Upload Coverage
  uses: codecov/codecov-action@v1
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the project root
2. **Missing Dependencies**: Run `python run_tests.py --check-deps`
3. **Permission Errors**: Check file permissions on test directory
4. **Memory Issues**: Use `--skip-slow` to avoid large dataset tests

### Debugging Failed Tests

```bash
# Run with verbose output and local variables
pytest -v --tb=long --showlocals test/test_failing_file.py

# Run just the failing test
pytest test/test_failing_file.py::TestClass::test_failing_method -v

# Stop on first failure for debugging
pytest -x test/
```

## Contributing

When adding new tests:

1. Follow the existing naming conventions
2. Use appropriate markers for categorization
3. Test both success and failure scenarios
4. Include edge cases and error handling
5. Add performance tests for new features
6. Update this README if adding new test files

## Test Statistics

- **Total Test Files**: 12
- **Test Classes**: 50+
- **Test Methods**: 500+
- **Coverage Target**: 100% of public API
- **Test Execution Time**: ~2-5 minutes (parallel)
- **Supported Python Versions**: 3.8+

## Best Practices

1. **Use Temporary Directories**: All tests use `tempfile.TemporaryDirectory()`
2. **Clean Resources**: Proper cleanup in `finally` blocks or context managers
3. **Test Isolation**: Each test is independent and can run alone
4. **Comprehensive Coverage**: Test happy paths, edge cases, and error conditions
5. **Performance Awareness**: Include timing assertions for critical operations
6. **Documentation**: Clear docstrings explaining test purpose and scenarios

This comprehensive test suite ensures the reliability, performance, and correctness of the ApexBase Python API across all supported use cases and configurations.
