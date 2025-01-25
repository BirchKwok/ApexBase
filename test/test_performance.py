from pathlib import Path
import pytest
import time
import random
import string
import psutil
import os
from apexbase import ApexClient
import numpy as np
import tempfile
import json
import shutil

@pytest.fixture
def temp_dir():
    """Create temporary directory"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)

def generate_random_string(length: int) -> str:
    """Generate random string"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def generate_test_records(count: int) -> list:
    """Generate test records"""
    records = []
    for _ in range(count):
        record = {
            "name": generate_random_string(10),
            "age": random.randint(18, 80),
            "email": f"{generate_random_string(8)}@example.com",
            "tags": [generate_random_string(5) for _ in range(random.randint(1, 5))],
            "address": {
                "city": generate_random_string(8),
                "street": generate_random_string(15),
                "number": random.randint(1, 1000)
            }
        }
        records.append(record)
    return records

def measure_memory() -> float:
    """Measure current process memory usage (MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_large_scale_performance(temp_dir):
    """Large-scale performance test"""
    print("\n=== Large-scale performance test started ===\n")
    
    client = ApexClient(temp_dir)
    
    # 1. Batch write performance test
    print("1. Batch write performance test")
    batch_sizes = [1000, 10000, 100000]
    
    for batch_size in batch_sizes:
        records = generate_test_records(batch_size)
        
        # Disable automatic FTS update to improve write performance
        client.set_auto_update_fts(False)
        
        start_time = time.time()
        ids = client.store(records)
        assert ids is not None
        assert len(ids) == batch_size
        end_time = time.time()
        
        print(f"Write {batch_size} records time: {end_time - start_time:.2f} seconds")
        print(f"Average write time per record: {(end_time - start_time) * 1000 / batch_size:.2f} milliseconds")
        
        # Manually rebuild FTS index
        print("Rebuilding FTS index...")
        start_time = time.time()
        client.rebuild_search_index()
        end_time = time.time()
        print(f"Rebuilding index time: {end_time - start_time:.2f} seconds")
        
        print()
    
    # 2. Query performance test
    print("\n2. Query performance test")
    
    # 2.1 Simple query
    print("2.1 Simple query performance")
    start_time = time.time()
    results = client.query("age > 30")
    end_time = time.time()
    print(f"Simple query time: {end_time - start_time:.2f} seconds")
    print(f"Returned record count: {len(results)}")
    
    # 2.2 Complex query
    print("\n2.2 Complex query performance")
    start_time = time.time()
    results = client.query("age > 30 AND age < 50")
    end_time = time.time()
    print(f"Complex query time: {end_time - start_time:.2f} seconds")
    print(f"Returned record count: {len(results)}")
    
    print("\n2.3 Full-text search performance")
    
    # Set fields to be searchable
    client.set_searchable("email", True)
    client.set_searchable("name", True)
    
    # Test different types of search
    search_terms = [
        "example",  # Simple word
        "john",     # Name
        "com"       # Domain part
    ]
    
    for term in search_terms:
        start_time = time.time()
        results = client.search_text(term)
        end_time = time.time()
        print(f"Search term '{term}' time: {end_time - start_time:.2f} seconds")
        print(f"Returned record count: {len(results)}")
    
    # 3. Batch retrieval performance test
    print("\n3. Batch retrieval performance test")
    sample_size = min(1000, len(results))
    sample_ids = random.sample(results.ids, sample_size)
    
    start_time = time.time()
    retrieved = client.retrieve_many(sample_ids)
    end_time = time.time()
    print(f"Batch retrieval {sample_size} records time: {end_time - start_time:.2f} seconds")
    print(f"Average retrieval time per record: {(end_time - start_time) * 1000 / sample_size:.2f} milliseconds")
    
    print("\n=== Large-scale performance test completed ===")

def test_data_integrity(temp_dir):
    """Data integrity test"""
    print("\n=== Data integrity test started ===\n")
    
    client = ApexClient(temp_dir)
    
    # 1. Generate test data
    print("1. Generate test data")
    total_records = 10000
    records = generate_test_records(total_records)
    
    # Record the hash value of the original data
    original_hashes = {}
    for record in records:
        record_str = json.dumps(record, sort_keys=True)
        original_hashes[hash(record_str)] = record
    
    # 2. Batch write test
    print("2. Batch write test")
    client.set_auto_update_fts(False)  # Disable FTS to improve performance
    ids = client.store(records)
    assert ids is not None
    assert len(ids) == total_records
    
    # 3. Data consistency verification
    print("3. Data consistency verification")
    retrieved_records = client.retrieve_many(ids)
    
    # Verify record count
    assert len(retrieved_records) == total_records, \
        f"Record count mismatch: Expected {total_records}, actual {len(retrieved_records)}"
    
    # Verify data content
    for record in retrieved_records:
        # Remove _id field before comparison
        record_copy = record.copy()
        del record_copy['_id']
        record_str = json.dumps(record_copy, sort_keys=True)
        record_hash = hash(record_str)
        
        assert record_hash in original_hashes, \
            f"Original data not found for record: {record}"
        
        original = original_hashes[record_hash]
        assert record_copy == original, \
            f"Data mismatch:\nOriginal: {original}\nActual: {record_copy}"
    
    print("Basic data integrity verification passed")
    
    # 4. Field type consistency test
    print("\n4. Field type consistency test")
    fields = client.list_fields()
    
    type_test_data = {
        "int_field": 42,
        "float_field": 3.14,
        "str_field": "test string",
        "bool_field": True,
        "list_field": [1, 2, 3],
        "dict_field": {"key": "value"},
        "null_field": None
    }
    
    # Store test data
    record_id = client.store(type_test_data)
    assert record_id is not None
    retrieved = client.retrieve(record_id)
    
    # Verify field types
    for field, value in type_test_data.items():
        if value is None:
            # SQLite ignores NULL values, so these fields may not exist in retrieval
            if field in retrieved:
                assert retrieved[field] is None, \
                    f"NULL value field should return None, actual {retrieved[field]}"
        else:
            assert field in retrieved, f"Field {field} lost"
            if field == "bool_field":
                assert isinstance(retrieved[field], int), \
                    f"Boolean field should be stored as an integer type, actual {type(retrieved[field])}"
                assert retrieved[field] in (0, 1), \
                    f"Boolean field value should be 0 or 1, actual {retrieved[field]}"
            else:
                assert type(retrieved[field]) == type(value), \
                    f"Type mismatch for field {field}: Expected {type(value)}, actual {type(retrieved[field])}"
    
    print("Field type consistency verification passed")
    
    # 5. Concurrent write consistency test
    print("\n5. Concurrent write consistency test")
    import threading
    
    concurrent_records = 1000
    threads_count = 4
    records_per_thread = concurrent_records // threads_count
    thread_results = {i: [] for i in range(threads_count)}
    
    def concurrent_write(thread_id):
        thread_records = generate_test_records(records_per_thread)
        thread_results[thread_id] = client.store(thread_records)
        assert thread_results[thread_id] is not None
        assert len(thread_results[thread_id]) == records_per_thread
    
    # Create and start threads
    threads = []
    for i in range(threads_count):
        t = threading.Thread(target=concurrent_write, args=(i,))
        threads.append(t)
        t.start()
    
    # Wait for all threads to complete
    for t in threads:
        t.join()
    
    # Verify all records were correctly stored
    all_ids = []
    for ids in thread_results.values():
        all_ids.extend(ids)
    
    retrieved = client.retrieve_many(all_ids)
    assert len(retrieved) == concurrent_records, \
        f"Concurrent write record count mismatch: Expected {concurrent_records}, actual {len(retrieved)}"
    
    print("Concurrent write consistency verification passed")
    
    # 6. Transaction consistency test
    print("\n6. Transaction consistency test")
    
    # Prepare update data
    update_data = {}
    for id_ in all_ids[:100]:
        update_data[id_] = {
            "name": "Updated " + generate_random_string(8),
            "age": random.randint(18, 80)
        }
    
    # Execute batch update
    success_ids = client.batch_replace(update_data)
    assert len(success_ids) == len(update_data), \
        f"Update record count mismatch: Expected {len(update_data)}, actual {len(success_ids)}"
    
    # Verify update results
    updated_records = client.retrieve_many(success_ids)
    for record in updated_records:
        expected = update_data[record['_id']]
        for field, value in expected.items():
            assert record[field] == value, \
                f"Updated value mismatch: Expected {value}, actual {record[field]}"
    
    print("Transaction consistency verification passed")
    
    print("\n=== Data integrity test completed ===")

if __name__ == "__main__":
    test_large_scale_performance(Path("."))
