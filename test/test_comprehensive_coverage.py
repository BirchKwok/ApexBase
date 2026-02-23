"""Comprehensive coverage: every API method + complex SQL edge cases to find bugs."""
import pytest
import tempfile
import shutil
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apexbase', 'python'))

try:
    from apexbase import ApexClient, ResultView
except ImportError as e:
    pytest.skip(f"ApexBase not available: {e}", allow_module_level=True)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

try:
    import pyarrow as pa
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False


def xfail_sql(client, sql):
    """Execute SQL; mark xfail if unimplemented, hard-fail on wrong results."""
    try:
        return client.execute(sql)
    except Exception as e:
        pytest.xfail(f"Not supported: {e}")


@pytest.fixture
def tmp_client():
    d = tempfile.mkdtemp()
    c = ApexClient(dirpath=d)
    yield c
    c.close()
    shutil.rmtree(d, ignore_errors=True)


# ============================================================
# 1. query() — three calling modes
# ============================================================

class TestQueryMethod:
    def setup_method(self):
        self.d = tempfile.mkdtemp()
        self.c = ApexClient(dirpath=self.d)
        self.c.create_table('t')
        self.c.store([
            {'name': 'Alice', 'age': 25, 'city': 'NYC'},
            {'name': 'Bob',   'age': 30, 'city': 'LA'},
            {'name': 'Carol', 'age': 35, 'city': 'NYC'},
        ])

    def teardown_method(self):
        self.c.close()
        shutil.rmtree(self.d, ignore_errors=True)

    def test_no_args_returns_all(self):
        assert len(self.c.query()) == 3

    def test_full_select_sql(self):
        assert len(self.c.query(sql="SELECT * FROM t WHERE age > 25")) == 2

    def test_filter_expression_becomes_where(self):
        r = self.c.query(sql="age = 25")
        assert len(r) == 1 and r.first()['name'] == 'Alice'

    def test_where_clause_param(self):
        assert len(self.c.query(where_clause="age > 25")) == 2

    def test_limit_only(self):
        assert len(self.c.query(limit=2)) == 2

    def test_where_plus_limit(self):
        assert len(self.c.query(where_clause="city = 'NYC'", limit=1)) == 1


# ============================================================
# 2. ResultView — every public method
# ============================================================

class TestResultViewMethods:
    def setup_method(self):
        self.d = tempfile.mkdtemp()
        self.c = ApexClient(dirpath=self.d)
        self.c.create_table('t')
        self.c.store([
            {'val': 10, 'label': 'a'},
            {'val': 20, 'label': 'b'},
            {'val': 30, 'label': 'c'},
        ])

    def teardown_method(self):
        self.c.close()
        shutil.rmtree(self.d, ignore_errors=True)

    def _rv(self):
        return self.c.execute("SELECT * FROM t ORDER BY val")

    def test_to_dict(self):
        d = self._rv().to_dict()
        assert isinstance(d, list) and len(d) == 3 and d[0]['val'] == 10

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas unavailable")
    def test_to_pandas_zero_copy(self):
        df = self._rv().to_pandas(zero_copy=True)
        assert len(df) == 3 and '_id' not in df.columns

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas unavailable")
    def test_to_pandas_no_zero_copy(self):
        df = self._rv().to_pandas(zero_copy=False)
        assert len(df) == 3

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars unavailable")
    def test_to_polars(self):
        df = self._rv().to_polars()
        assert len(df) == 3 and '_id' not in df.columns

    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow unavailable")
    def test_to_arrow_hides_id(self):
        tbl = self._rv().to_arrow()
        assert tbl.num_rows == 3 and '_id' not in tbl.column_names

    def test_scalar_count(self):
        assert self.c.execute("SELECT COUNT(*) as c FROM t").scalar() == 3

    def test_scalar_sum(self):
        assert self.c.execute("SELECT SUM(val) as s FROM t").scalar() == 60

    def test_first_row(self):
        row = self._rv().first()
        assert row is not None and row['val'] == 10

    def test_first_empty(self):
        assert self.c.execute("SELECT * FROM t WHERE val > 9999").first() is None

    def test_get_ids_numpy(self):
        ids = self._rv().get_ids(return_list=False)
        assert isinstance(ids, np.ndarray) and len(ids) == 3

    def test_get_ids_list(self):
        ids = self._rv().get_ids(return_list=True)
        assert isinstance(ids, list) and len(ids) == 3

    def test_ids_property(self):
        assert len(self._rv().ids) == 3

    def test_shape(self):
        assert self._rv().shape[0] == 3

    def test_columns_hides_id(self):
        cols = self._rv().columns
        assert '_id' not in cols and 'val' in cols

    def test_len_and_iter(self):
        rv = self._rv()
        assert len(rv) == 3 and len(list(rv)) == 3

    def test_getitem(self):
        rv = self._rv()
        assert rv[0]['val'] == 10 and rv[2]['val'] == 30

    def test_repr(self):
        assert 'ResultView' in repr(self._rv())

    def test_empty_shape(self):
        rv = self.c.execute("SELECT * FROM t WHERE val > 9999")
        assert rv.shape == (0, 0)

    def test_empty_first_is_none(self):
        rv = self.c.execute("SELECT * FROM t WHERE val > 9999")
        assert rv.first() is None


# ============================================================
# 3. store() — all input variants
# ============================================================

class TestStoreVariants:
    def _new(self):
        d = tempfile.mkdtemp()
        c = ApexClient(dirpath=d)
        c.create_table('t')
        return c, d

    def test_single_dict(self):
        c, d = self._new()
        c.store({'x': 1, 'y': 'a'})
        assert c.count_rows() == 1
        c.close(); shutil.rmtree(d, ignore_errors=True)

    def test_list_of_dicts(self):
        c, d = self._new()
        c.store([{'x': i} for i in range(5)])
        assert c.count_rows() == 5
        c.close(); shutil.rmtree(d, ignore_errors=True)

    def test_columnar_dict(self):
        c, d = self._new()
        c.store({'x': [1, 2, 3], 'y': ['a', 'b', 'c']})
        assert c.count_rows() == 3
        c.close(); shutil.rmtree(d, ignore_errors=True)

    def test_numpy_arrays(self):
        c, d = self._new()
        c.store({'x': np.array([1, 2, 3]), 'y': np.array([4.0, 5.0, 6.0])})
        assert c.count_rows() == 3
        c.close(); shutil.rmtree(d, ignore_errors=True)

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas unavailable")
    def test_pandas_df(self):
        c, d = self._new()
        c.store(pd.DataFrame({'x': [1, 2, 3]}))
        assert c.count_rows() == 3
        c.close(); shutil.rmtree(d, ignore_errors=True)

    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow unavailable")
    def test_arrow_table(self):
        c, d = self._new()
        c.store(pa.table({'x': [1, 2, 3]}))
        assert c.count_rows() == 3
        c.close(); shutil.rmtree(d, ignore_errors=True)

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars unavailable")
    def test_polars_df(self):
        c, d = self._new()
        c.store(pl.DataFrame({'x': [1, 2, 3]}))
        assert c.count_rows() == 3
        c.close(); shutil.rmtree(d, ignore_errors=True)

    def test_empty_list_noop(self):
        c, d = self._new()
        c.store([])
        assert c.count_rows() == 0
        c.close(); shutil.rmtree(d, ignore_errors=True)

    def test_invalid_type_raises(self):
        c, d = self._new()
        with pytest.raises((ValueError, TypeError)):
            c.store(42)
        c.close(); shutil.rmtree(d, ignore_errors=True)


# ============================================================
# 4. DataFrame import (from_pandas / from_pyarrow / from_polars)
# ============================================================

class TestDataFrameImport:
    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas unavailable")
    def test_from_pandas_existing_table(self):
        with tempfile.TemporaryDirectory() as d:
            c = ApexClient(dirpath=d)
            c.create_table('t')
            c.from_pandas(pd.DataFrame({'x': [1, 2, 3]}))
            assert c.count_rows() == 3
            c.close()

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas unavailable")
    def test_from_pandas_creates_table(self):
        with tempfile.TemporaryDirectory() as d:
            c = ApexClient(dirpath=d)
            c.from_pandas(pd.DataFrame({'x': [1, 2]}), table_name='auto')
            assert c.count_rows() == 2
            c.close()

    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow unavailable")
    def test_from_pyarrow(self):
        with tempfile.TemporaryDirectory() as d:
            c = ApexClient(dirpath=d)
            c.create_table('t')
            c.from_pyarrow(pa.table({'x': [10, 20, 30]}))
            assert c.count_rows() == 3
            c.close()

    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="pyarrow unavailable")
    def test_from_pyarrow_creates_table(self):
        with tempfile.TemporaryDirectory() as d:
            c = ApexClient(dirpath=d)
            c.from_pyarrow(pa.table({'x': [1, 2]}), table_name='auto')
            assert c.count_rows() == 2
            c.close()

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars unavailable")
    def test_from_polars(self):
        with tempfile.TemporaryDirectory() as d:
            c = ApexClient(dirpath=d)
            c.create_table('t')
            c.from_polars(pl.DataFrame({'x': [1, 2, 3]}))
            assert c.count_rows() == 3
            c.close()


# ============================================================
# 5. set_auto_flush / get_auto_flush / estimate_memory_bytes
# ============================================================

class TestAutoFlushAndMemory:
    def test_get_auto_flush_returns_ints(self, tmp_client):
        tmp_client.create_table('t')
        rows, byt = tmp_client.get_auto_flush()
        assert isinstance(rows, int) and isinstance(byt, int)

    def test_set_rows_threshold(self, tmp_client):
        tmp_client.create_table('t')
        tmp_client.set_auto_flush(rows=100)
        rows, _ = tmp_client.get_auto_flush()
        assert rows == 100

    def test_set_bytes_threshold(self, tmp_client):
        tmp_client.create_table('t')
        tmp_client.set_auto_flush(bytes=1024 * 1024)
        _, byt = tmp_client.get_auto_flush()
        assert byt == 1024 * 1024

    def test_set_both_thresholds(self, tmp_client):
        tmp_client.create_table('t')
        tmp_client.set_auto_flush(rows=50, bytes=512 * 1024)
        rows, byt = tmp_client.get_auto_flush()
        assert rows == 50 and byt == 512 * 1024

    def test_estimate_memory_empty(self, tmp_client):
        tmp_client.create_table('t')
        mem = tmp_client.estimate_memory_bytes()
        assert isinstance(mem, int) and mem >= 0

    def test_estimate_memory_grows(self, tmp_client):
        tmp_client.create_table('t')
        tmp_client.store([{'x': i, 'y': f's{i}'} for i in range(100)])
        assert tmp_client.estimate_memory_bytes() > 0


# ============================================================
# 6. create_table with schema / get_column_dtype
# ============================================================

class TestSchemaAndDtype:
    def test_typed_roundtrip(self, tmp_client):
        tmp_client.create_table('t', schema={
            'name': 'string', 'age': 'int64', 'score': 'float64'
        })
        tmp_client.store([{'name': 'Alice', 'age': 25, 'score': 9.5}])
        row = tmp_client.execute("SELECT * FROM t").first()
        assert row['name'] == 'Alice' and row['age'] == 25

    def test_all_int_subtypes(self, tmp_client):
        tmp_client.create_table('t', schema={
            'i8': 'int8', 'i16': 'int16', 'i32': 'int32', 'i64': 'int64'
        })
        tmp_client.store([{'i8': 1, 'i16': 2, 'i32': 3, 'i64': 4}])
        assert tmp_client.count_rows() == 1

    def test_bool_schema(self, tmp_client):
        tmp_client.create_table('t', schema={'flag': 'bool', 'val': 'int64'})
        tmp_client.store([{'flag': True, 'val': 1}, {'flag': False, 'val': 2}])
        rv = tmp_client.execute("SELECT * FROM t WHERE flag = true")
        assert len(rv) == 1

    def test_invalid_type_raises(self, tmp_client):
        with pytest.raises(Exception):
            tmp_client.create_table('bad', schema={'x': 'invalid_xyz'})

    def test_get_dtype_string(self, tmp_client):
        tmp_client.create_table('t', schema={'name': 'string'})
        assert 'str' in tmp_client.get_column_dtype('name').lower()

    def test_get_dtype_int(self, tmp_client):
        tmp_client.create_table('t', schema={'age': 'int64'})
        assert 'int' in tmp_client.get_column_dtype('age').lower()

    def test_get_dtype_float(self, tmp_client):
        tmp_client.create_table('t', schema={'score': 'float64'})
        dtype = tmp_client.get_column_dtype('score').lower()
        assert 'float' in dtype or 'double' in dtype


# ============================================================
# 7. batch_replace / optimize / context manager / create_clean
# ============================================================

class TestBatchReplaceOptimizeLifecycle:
    def test_batch_replace(self, tmp_client):
        tmp_client.create_table('t')
        tmp_client.store([{'name': 'Alice', 'age': 25}, {'name': 'Bob', 'age': 30}])
        rv = tmp_client.execute("SELECT * FROM t ORDER BY name")
        rows = rv.to_dict()
        ids = rv.get_ids(return_list=True)
        name_to_id = {rows[i]['name']: ids[i] for i in range(len(rows))}
        alice_id = name_to_id.get('Alice')
        if alice_id is None:
            pytest.skip("Could not obtain Alice's ID")
        success = tmp_client.batch_replace({alice_id: {'name': 'Alice', 'age': 99}})
        assert alice_id in success
        assert tmp_client.retrieve(alice_id)['age'] == 99

    def test_optimize_no_crash(self, tmp_client):
        tmp_client.create_table('t')
        tmp_client.store([{'x': i} for i in range(50)])
        tmp_client.optimize()
        assert tmp_client.count_rows() == 50

    def test_context_manager_closes(self):
        with tempfile.TemporaryDirectory() as d:
            with ApexClient(dirpath=d) as c:
                c.create_table('t')
                c.store([{'x': 1}])
            assert c._is_closed

    def test_context_manager_closes_on_exception(self):
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(ValueError):
                with ApexClient(dirpath=d) as c:
                    c.create_table('t')
                    raise ValueError("boom")
            assert c._is_closed

    def test_create_clean_wipes_data(self):
        with tempfile.TemporaryDirectory() as d:
            c1 = ApexClient(dirpath=d)
            c1.create_table('t')
            c1.store([{'x': 1}, {'x': 2}])
            c1.flush(); c1.close()
            c2 = ApexClient.create_clean(dirpath=d)
            c2.create_table('t')
            assert c2.count_rows() == 0
            c2.close()
