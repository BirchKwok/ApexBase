"""
ApexBase vs DuckDB Performance Benchmark

Compares ApexBase performance against DuckDB on identical workloads.
Results include test environment details for reproducibility.
"""

import time
import tempfile
import os
import platform
import subprocess
from typing import Dict, List, Tuple

# Test configuration
N_ROWS = 1_000_000
N_ITERATIONS = 5
WARMUP_ITERATIONS = 2


def get_system_info() -> Dict[str, str]:
    """Get detailed system information for benchmark report."""
    info = {
        "Platform": platform.platform(),
        "Processor": platform.processor(),
        "Machine": platform.machine(),
        "Python Version": platform.python_version(),
    }
    
    # Try to get CPU info on macOS
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], 
                                   capture_output=True, text=True)
            if result.returncode == 0:
                info["CPU"] = result.stdout.strip()
            
            # Get memory info
            result = subprocess.run(["sysctl", "-n", "hw.memsize"], 
                                   capture_output=True, text=True)
            if result.returncode == 0:
                mem_bytes = int(result.stdout.strip())
                info["Memory"] = f"{mem_bytes / (1024**3):.1f} GB"
    except Exception:
        pass
    
    return info


def benchmark_apexbase() -> Dict[str, List[float]]:
    """Benchmark ApexBase performance."""
    import apexbase
    from apexbase import ApexClient
    
    results = {}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        client = ApexClient(os.path.join(tmpdir, "benchmark"))
        client.create_table("users")
        
        # Generate test data
        print("Generating test data...")
        data = [
            {
                "name": f"user_{i % 10000}",
                "age": i % 100,
                "score": float(i % 1000) / 10.0,
                "category": f"cat_{i % 10}"
            }
            for i in range(N_ROWS)
        ]
        
        # Insert data
        start = time.perf_counter()
        client.store(data)
        insert_time = (time.perf_counter() - start) * 1000
        results["Insert 1M rows"] = [insert_time]
        
        # Define test queries
        queries = [
            ("SELECT COUNT(*) FROM users", "COUNT(*)"),
            ("SELECT * FROM users LIMIT 100", "SELECT * LIMIT 100"),
            ("SELECT * FROM users LIMIT 10000", "SELECT * LIMIT 10K"),
            ("SELECT age, COUNT(*) FROM users GROUP BY age", "GROUP BY age (100 groups)"),
            ("SELECT name, age, COUNT(*) FROM users GROUP BY name, age", "GROUP BY name, age (10K groups)"),
            ("SELECT age FROM users ORDER BY age LIMIT 100", "ORDER BY + LIMIT 100"),
            ("SELECT * FROM users ORDER BY age LIMIT 1000", "ORDER BY + LIMIT 1K"),
            ("SELECT * FROM users WHERE age > 50", "Filter (age > 50)"),
            ("SELECT * FROM users WHERE name = 'user_5000'", "Filter (name = 'user_5000')"),
        ]
        
        # Warmup
        for _ in range(WARMUP_ITERATIONS):
            for sql, _ in queries:
                try:
                    client.execute(sql)
                except Exception:
                    pass
        
        # Benchmark each query
        for sql, name in queries:
            times = []
            for _ in range(N_ITERATIONS):
                start = time.perf_counter()
                try:
                    result = client.execute(sql)
                    # Materialize result to ensure complete execution
                    _ = len(result)
                except Exception as e:
                    print(f"  Error in {name}: {e}")
                    times.append(None)
                    continue
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
            
            # Filter out None values (errors)
            valid_times = [t for t in times if t is not None]
            if valid_times:
                results[name] = valid_times
        
        client.close()
    
    return results


def benchmark_duckdb() -> Dict[str, List[float]]:
    """Benchmark DuckDB performance."""
    try:
        import duckdb
    except ImportError:
        print("DuckDB not installed. Run: pip install duckdb")
        return {}
    
    results = {}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "benchmark.duckdb")
        conn = duckdb.connect(db_path)
        
        # Create table
        conn.execute("""
            CREATE TABLE users (
                name VARCHAR,
                age INTEGER,
                score DOUBLE,
                category VARCHAR
            )
        """)
        
        # Generate and insert data
        print("Generating test data for DuckDB...")
        data = [
            (f"user_{i % 10000}", i % 100, float(i % 1000) / 10.0, f"cat_{i % 10}")
            for i in range(N_ROWS)
        ]
        
        start = time.perf_counter()
        conn.executemany("INSERT INTO users VALUES (?, ?, ?, ?)", data)
        insert_time = (time.perf_counter() - start) * 1000
        results["Insert 1M rows"] = [insert_time]
        
        # Define test queries (DuckDB syntax)
        queries = [
            ("SELECT COUNT(*) FROM users", "COUNT(*)"),
            ("SELECT * FROM users LIMIT 100", "SELECT * LIMIT 100"),
            ("SELECT * FROM users LIMIT 10000", "SELECT * LIMIT 10K"),
            ("SELECT age, COUNT(*) FROM users GROUP BY age", "GROUP BY age (100 groups)"),
            ("SELECT name, age, COUNT(*) FROM users GROUP BY name, age", "GROUP BY name, age (10K groups)"),
            ("SELECT age FROM users ORDER BY age LIMIT 100", "ORDER BY + LIMIT 100"),
            ("SELECT * FROM users ORDER BY age LIMIT 1000", "ORDER BY + LIMIT 1K"),
            ("SELECT * FROM users WHERE age > 50", "Filter (age > 50)"),
            ("SELECT * FROM users WHERE name = 'user_5000'", "Filter (name = 'user_5000')"),
        ]
        
        # Warmup
        for _ in range(WARMUP_ITERATIONS):
            for sql, _ in queries:
                try:
                    conn.execute(sql).fetchall()
                except Exception:
                    pass
        
        # Benchmark each query
        for sql, name in queries:
            times = []
            for _ in range(N_ITERATIONS):
                start = time.perf_counter()
                try:
                    result = conn.execute(sql).fetchall()
                except Exception as e:
                    print(f"  Error in {name}: {e}")
                    times.append(None)
                    continue
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
            
            valid_times = [t for t in times if t is not None]
            if valid_times:
                results[name] = valid_times
        
        conn.close()
    
    return results


def format_results(apex_results: Dict, duck_results: Dict) -> str:
    """Format benchmark results as markdown table."""
    lines = []
    lines.append("## Performance Comparison: ApexBase vs DuckDB\n")
    lines.append(f"**Dataset**: {N_ROWS:,} rows")
    lines.append(f"**Iterations**: {N_ITERATIONS} (after {WARMUP_ITERATIONS} warmup)\n")
    
    lines.append("| Query | ApexBase | DuckDB | Ratio |")
    lines.append("|-------|----------|--------|-------|")
    
    # Get all unique test names
    all_tests = set(apex_results.keys()) | set(duck_results.keys())
    
    for test in sorted(all_tests):
        apex_times = apex_results.get(test, [])
        duck_times = duck_results.get(test, [])
        
        if apex_times and duck_times:
            apex_avg = sum(apex_times) / len(apex_times)
            duck_avg = sum(duck_times) / len(duck_times)
            ratio = apex_avg / duck_avg if duck_avg > 0 else float('inf')
            
            apex_str = f"{apex_avg:.2f}ms"
            duck_str = f"{duck_avg:.2f}ms"
            ratio_str = f"{ratio:.2f}x"
            
            lines.append(f"| {test} | {apex_str} | {duck_str} | {ratio_str} |")
        elif apex_times:
            apex_avg = sum(apex_times) / len(apex_times)
            lines.append(f"| {test} | {apex_avg:.2f}ms | N/A | - |")
        elif duck_times:
            duck_avg = sum(duck_times) / len(duck_times)
            lines.append(f"| {test} | N/A | {duck_avg:.2f}ms | - |")
    
    return "\n".join(lines)


def print_environment_info():
    """Print system environment information."""
    info = get_system_info()
    
    print("\n" + "="*60)
    print("TEST ENVIRONMENT")
    print("="*60)
    for key, value in info.items():
        print(f"{key:20}: {value}")
    
    # Get package versions
    try:
        import apexbase
        print(f"{'ApexBase Version':20}: {apexbase.__version__}")
    except Exception:
        pass
    
    try:
        import duckdb
        print(f"{'DuckDB Version':20}: {duckdb.__version__}")
    except Exception:
        pass
    
    try:
        import pyarrow as pa
        print(f"{'PyArrow Version':20}: {pa.__version__}")
    except Exception:
        pass
    
    print("="*60 + "\n")


def main():
    print_environment_info()
    
    print("Running ApexBase benchmarks...")
    apex_results = benchmark_apexbase()
    
    print("\nRunning DuckDB benchmarks...")
    duck_results = benchmark_duckdb()
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    report = format_results(apex_results, duck_results)
    print(report)
    
    # Also save to file
    with open("benchmark_vs_duckdb_report.md", "w") as f:
        f.write("# ApexBase vs DuckDB Benchmark Report\n\n")
        
        # Add environment info
        info = get_system_info()
        f.write("## Test Environment\n\n")
        for key, value in info.items():
            f.write(f"- **{key}**: {value}\n")
        
        try:
            import apexbase
            f.write(f"- **ApexBase Version**: {apexbase.__version__}\n")
        except Exception:
            pass
        
        try:
            import duckdb
            f.write(f"- **DuckDB Version**: {duckdb.__version__}\n")
        except Exception:
            pass
        
        f.write("\n")
        f.write(report)
        f.write("\n\n")
        f.write("## Notes\n\n")
        f.write("- Lower time is better\n")
        f.write("- Ratio > 1 means ApexBase is slower than DuckDB\n")
        f.write("- Ratio < 1 means ApexBase is faster than DuckDB\n")
    
    print("\nReport saved to: benchmark_vs_duckdb_report.md")


if __name__ == "__main__":
    main()
