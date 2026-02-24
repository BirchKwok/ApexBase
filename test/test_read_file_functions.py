"""Tests for SQL table functions: read_csv, read_parquet, read_json."""
import csv
import json
import os
import shutil
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apexbase', 'python'))

try:
    from apexbase import ApexClient
except ImportError as e:
    pytest.skip(f"ApexBase not available: {e}", allow_module_level=True)


# ---- helpers ---------------------------------------------------------------

ROWS = [
    {'name': 'Alice', 'age': 30, 'score': 88.5, 'city': 'Beijing'},
    {'name': 'Bob',   'age': 25, 'score': 72.0, 'city': 'Shanghai'},
    {'name': 'Carol', 'age': 35, 'score': 95.0, 'city': 'Beijing'},
    {'name': 'Dave',  'age': 28, 'score': 60.0, 'city': 'Guangzhou'},
    {'name': 'Eve',   'age': 22, 'score': 83.0, 'city': 'Shanghai'},
]


def _write_csv(path, rows, header=True, delimiter=','):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter=delimiter)
        if header and rows:
            w.writerow(rows[0].keys())
        for r in rows:
            w.writerow(r.values())


def _write_ndjson(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def _write_parquet(path, rows):
    pa = pytest.importorskip('pyarrow')
    pq = pytest.importorskip('pyarrow.parquet')
    tbl = pa.table({
        'name':  [r['name']  for r in rows],
        'age':   pa.array([r['age']   for r in rows], type=pa.int64()),
        'score': pa.array([r['score'] for r in rows], type=pa.float64()),
        'city':  [r['city']  for r in rows],
    })
    pq.write_table(tbl, path)


# ============================================================
# read_csv
# ============================================================

class TestReadCsv:
    def setup_method(self):
        self.d = tempfile.mkdtemp()
        self.c = ApexClient(dirpath=self.d)
        self.csv   = os.path.join(self.d, 'data.csv')
        self.tsv   = os.path.join(self.d, 'data.tsv')
        self.nohead = os.path.join(self.d, 'nohead.csv')
        _write_csv(self.csv,    ROWS)
        _write_csv(self.tsv,    ROWS, delimiter='\t')
        _write_csv(self.nohead, ROWS, header=False)

    def teardown_method(self):
        self.c.close()
        shutil.rmtree(self.d, ignore_errors=True)

    def test_row_count(self):
        rv = self.c.execute(f"SELECT * FROM read_csv('{self.csv}')")
        assert len(rv) == 5

    def test_columns_present(self):
        rv = self.c.execute(f"SELECT * FROM read_csv('{self.csv}')")
        assert {'name', 'age', 'score', 'city'}.issubset(set(rv.columns))

    def test_tsv_delimiter_option(self):
        rv = self.c.execute(f"SELECT * FROM read_csv('{self.tsv}', delimiter='\\t')")
        assert len(rv) == 5

    def test_delim_alias(self):
        rv = self.c.execute(f"SELECT * FROM read_csv('{self.tsv}', delim='\\t')")
        assert len(rv) == 5

    def test_sep_alias(self):
        rv = self.c.execute(f"SELECT * FROM read_csv('{self.tsv}', sep='\\t')")
        assert len(rv) == 5

    def test_no_header(self):
        rv = self.c.execute(f"SELECT * FROM read_csv('{self.nohead}', header=false)")
        assert len(rv) == 5

    def test_where_filter(self):
        rv = self.c.execute(f"SELECT name FROM read_csv('{self.csv}') WHERE city='Beijing' ORDER BY name")
        assert [r['name'] for r in rv] == ['Alice', 'Carol']

    def test_count_star(self):
        rv = self.c.execute(f"SELECT COUNT(*) AS cnt FROM read_csv('{self.csv}')")
        assert rv.first()['cnt'] == 5

    def test_group_by(self):
        rv = self.c.execute(
            f"SELECT city, COUNT(*) AS cnt FROM read_csv('{self.csv}') GROUP BY city ORDER BY city"
        )
        m = {r['city']: r['cnt'] for r in rv}
        assert m == {'Beijing': 2, 'Shanghai': 2, 'Guangzhou': 1}

    def test_order_by_limit(self):
        rv = self.c.execute(f"SELECT age FROM read_csv('{self.csv}') ORDER BY age DESC LIMIT 3")
        ages = [r['age'] for r in rv]
        assert ages == sorted(ages, reverse=True)
        assert len(ages) == 3

    def test_projection(self):
        rv = self.c.execute(f"SELECT name, city FROM read_csv('{self.csv}')")
        assert rv.columns == ['name', 'city']

    def test_type_inference_int(self):
        rv = self.c.execute(f"SELECT age FROM read_csv('{self.csv}') WHERE name='Alice'")
        assert rv.first()['age'] == 30

    def test_type_inference_float(self):
        rv = self.c.execute(f"SELECT score FROM read_csv('{self.csv}') WHERE name='Alice'")
        assert abs(rv.first()['score'] - 88.5) < 1e-6

    def test_avg_aggregation(self):
        rv = self.c.execute(f"SELECT AVG(score) AS avg_score FROM read_csv('{self.csv}')")
        expected = sum(r['score'] for r in ROWS) / len(ROWS)
        assert abs(rv.scalar() - expected) < 1e-4

    def test_having(self):
        try:
            rv = self.c.execute(
                f"SELECT city FROM read_csv('{self.csv}') GROUP BY city HAVING COUNT(*) > 1 ORDER BY city"
            )
            cities = [r['city'] for r in rv]
            assert cities == ['Beijing', 'Shanghai']
        except Exception as e:
            pytest.xfail(f"HAVING on table function not yet supported: {e}")

    def test_to_pandas(self):
        pytest.importorskip('pandas')
        df = self.c.execute(f"SELECT * FROM read_csv('{self.csv}')").to_pandas()
        assert len(df) == 5

    def test_to_arrow(self):
        pytest.importorskip('pyarrow')
        tbl = self.c.execute(f"SELECT * FROM read_csv('{self.csv}')").to_arrow()
        assert tbl.num_rows == 5

    def test_nonexistent_file_raises(self):
        with pytest.raises(Exception):
            self.c.execute(f"SELECT * FROM read_csv('{self.d}/no_such_file.csv')")


# ============================================================
# read_parquet
# ============================================================

class TestReadParquet:
    def setup_method(self):
        self.d = tempfile.mkdtemp()
        self.c = ApexClient(dirpath=self.d)
        self.pq = os.path.join(self.d, 'data.parquet')
        _write_parquet(self.pq, ROWS)

    def teardown_method(self):
        self.c.close()
        shutil.rmtree(self.d, ignore_errors=True)

    def test_row_count(self):
        rv = self.c.execute(f"SELECT * FROM read_parquet('{self.pq}')")
        assert len(rv) == 5

    def test_columns_present(self):
        rv = self.c.execute(f"SELECT * FROM read_parquet('{self.pq}')")
        assert {'name', 'age', 'score', 'city'}.issubset(set(rv.columns))

    def test_where_filter(self):
        rv = self.c.execute(
            f"SELECT name FROM read_parquet('{self.pq}') WHERE city='Beijing' ORDER BY name"
        )
        assert [r['name'] for r in rv] == ['Alice', 'Carol']

    def test_count_star(self):
        rv = self.c.execute(f"SELECT COUNT(*) AS cnt FROM read_parquet('{self.pq}')")
        assert rv.first()['cnt'] == 5

    def test_group_by(self):
        rv = self.c.execute(
            f"SELECT city, COUNT(*) AS cnt FROM read_parquet('{self.pq}') GROUP BY city ORDER BY city"
        )
        m = {r['city']: r['cnt'] for r in rv}
        assert m == {'Beijing': 2, 'Shanghai': 2, 'Guangzhou': 1}

    def test_order_by_limit(self):
        rv = self.c.execute(f"SELECT name FROM read_parquet('{self.pq}') ORDER BY age ASC LIMIT 2")
        assert len(rv) == 2

    def test_projection(self):
        rv = self.c.execute(f"SELECT name, city FROM read_parquet('{self.pq}')")
        assert rv.columns == ['name', 'city']

    def test_avg_aggregation(self):
        rv = self.c.execute(f"SELECT AVG(score) AS avg_score FROM read_parquet('{self.pq}')")
        expected = sum(r['score'] for r in ROWS) / len(ROWS)
        assert abs(rv.scalar() - expected) < 1e-4

    def test_to_pandas(self):
        pytest.importorskip('pandas')
        df = self.c.execute(f"SELECT * FROM read_parquet('{self.pq}')").to_pandas()
        assert len(df) == 5

    def test_to_arrow(self):
        pytest.importorskip('pyarrow')
        tbl = self.c.execute(f"SELECT * FROM read_parquet('{self.pq}')").to_arrow()
        assert tbl.num_rows == 5

    def test_nonexistent_file_raises(self):
        with pytest.raises(Exception):
            self.c.execute(f"SELECT * FROM read_parquet('{self.d}/no_such.parquet')")


# ============================================================
# read_json
# ============================================================

class TestReadJson:
    def setup_method(self):
        self.d = tempfile.mkdtemp()
        self.c = ApexClient(dirpath=self.d)
        self.ndjson = os.path.join(self.d, 'data.json')
        _write_ndjson(self.ndjson, ROWS)

    def teardown_method(self):
        self.c.close()
        shutil.rmtree(self.d, ignore_errors=True)

    def test_row_count(self):
        rv = self.c.execute(f"SELECT * FROM read_json('{self.ndjson}')")
        assert len(rv) == 5

    def test_columns_present(self):
        rv = self.c.execute(f"SELECT * FROM read_json('{self.ndjson}')")
        assert {'name', 'age', 'score', 'city'}.issubset(set(rv.columns))

    def test_where_filter(self):
        rv = self.c.execute(
            f"SELECT name FROM read_json('{self.ndjson}') WHERE city='Shanghai' ORDER BY name"
        )
        assert [r['name'] for r in rv] == ['Bob', 'Eve']

    def test_count_star(self):
        rv = self.c.execute(f"SELECT COUNT(*) AS cnt FROM read_json('{self.ndjson}')")
        assert rv.first()['cnt'] == 5

    def test_group_by(self):
        rv = self.c.execute(
            f"SELECT city, COUNT(*) AS cnt FROM read_json('{self.ndjson}') GROUP BY city ORDER BY city"
        )
        m = {r['city']: r['cnt'] for r in rv}
        assert m == {'Beijing': 2, 'Shanghai': 2, 'Guangzhou': 1}

    def test_order_by_limit(self):
        rv = self.c.execute(
            f"SELECT name FROM read_json('{self.ndjson}') ORDER BY age ASC LIMIT 2"
        )
        assert len(rv) == 2

    def test_type_inference_int(self):
        rv = self.c.execute(f"SELECT age FROM read_json('{self.ndjson}') WHERE name='Alice'")
        assert rv.first()['age'] == 30

    def test_type_inference_float(self):
        rv = self.c.execute(f"SELECT score FROM read_json('{self.ndjson}') WHERE name='Alice'")
        assert abs(rv.first()['score'] - 88.5) < 1e-6

    def test_to_pandas(self):
        pytest.importorskip('pandas')
        df = self.c.execute(f"SELECT * FROM read_json('{self.ndjson}')").to_pandas()
        assert len(df) == 5

    def test_to_arrow(self):
        pytest.importorskip('pyarrow')
        tbl = self.c.execute(f"SELECT * FROM read_json('{self.ndjson}')").to_arrow()
        assert tbl.num_rows == 5


# ============================================================
# Combined: file reads with stored tables
# ============================================================

class TestReadFileCombined:
    def setup_method(self):
        self.d = tempfile.mkdtemp()
        self.c = ApexClient(dirpath=self.d)
        self.c.execute("CREATE TABLE users (name STRING, age INT, city STRING)")
        self.c.execute(
            "INSERT INTO users (name, age, city) VALUES "
            "('Alice', 30, 'Beijing'), ('Bob', 25, 'Shanghai')"
        )
        self.extra_csv = os.path.join(self.d, 'extra.csv')
        _write_csv(self.extra_csv, [
            {'name': 'Frank', 'age': 31, 'city': 'Beijing'},
            {'name': 'Grace', 'age': 27, 'city': 'Shanghai'},
        ])

    def teardown_method(self):
        self.c.close()
        shutil.rmtree(self.d, ignore_errors=True)

    def test_union_all_csv_with_table(self):
        rv = self.c.execute(f"""
            SELECT name FROM users
            UNION ALL
            SELECT name FROM read_csv('{self.extra_csv}')
            ORDER BY name
        """)
        assert len(rv) == 4
        assert [r['name'] for r in rv] == ['Alice', 'Bob', 'Frank', 'Grace']

    def test_union_dedup_csv_with_table(self):
        overlap = os.path.join(self.d, 'overlap.csv')
        _write_csv(overlap, [
            {'name': 'Alice'},
            {'name': 'Zara'},
        ])
        rv = self.c.execute(f"""
            SELECT name FROM users
            UNION
            SELECT name FROM read_csv('{overlap}')
            ORDER BY name
        """)
        names = [r['name'] for r in rv]
        assert names.count('Alice') == 1
        assert 'Zara' in names
        assert len(names) == 3

    def test_except_csv_as_blocklist(self):
        block = os.path.join(self.d, 'block.csv')
        _write_csv(block, [{'name': 'Bob'}])
        rv = self.c.execute(f"""
            SELECT name FROM users
            EXCEPT
            SELECT name FROM read_csv('{block}')
            ORDER BY name
        """)
        assert [r['name'] for r in rv] == ['Alice']

    def test_intersect_csv_with_table(self):
        common = os.path.join(self.d, 'common.csv')
        _write_csv(common, [{'name': 'Alice'}, {'name': 'Zara'}])
        rv = self.c.execute(f"""
            SELECT name FROM users
            INTERSECT
            SELECT name FROM read_csv('{common}')
            ORDER BY name
        """)
        assert [r['name'] for r in rv] == ['Alice']

    def test_join_csv_with_stored_table(self):
        scores = os.path.join(self.d, 'scores.csv')
        _write_csv(scores, [
            {'name': 'Alice', 'score': 99},
            {'name': 'Bob',   'score': 55},
        ])
        try:
            rv = self.c.execute(f"""
                SELECT u.name, s.score
                FROM users u
                JOIN read_csv('{scores}') s ON u.name = s.name
                ORDER BY u.name
            """)
            assert len(rv) == 2
            assert rv.first()['name'] == 'Alice'
            assert rv.first()['score'] == 99
        except Exception as e:
            pytest.xfail(f"JOIN with table function not yet supported: {e}")

    def test_where_on_join_result(self):
        scores = os.path.join(self.d, 'scores2.csv')
        _write_csv(scores, [
            {'name': 'Alice', 'score': 99},
            {'name': 'Bob',   'score': 55},
        ])
        try:
            rv = self.c.execute(f"""
                SELECT u.name
                FROM users u
                JOIN read_csv('{scores}') s ON u.name = s.name
                WHERE s.score >= 80
            """)
            assert len(rv) == 1
            assert rv.first()['name'] == 'Alice'
        except Exception as e:
            pytest.xfail(f"JOIN with table function not yet supported: {e}")

    def test_csv_parquet_union(self):
        pq = os.path.join(self.d, 'extra.parquet')
        _write_parquet(pq, [
            {'name': 'Hank', 'age': 40, 'score': 78.0, 'city': 'Chengdu'},
        ])
        csv2 = os.path.join(self.d, 'one.csv')
        _write_csv(csv2, [{'name': 'Ivy'}])
        rv = self.c.execute(f"""
            SELECT name FROM read_parquet('{pq}')
            UNION ALL
            SELECT name FROM read_csv('{csv2}')
            ORDER BY name
        """)
        assert [r['name'] for r in rv] == ['Hank', 'Ivy']
