"""P0-5: Constraint system tests (NOT NULL / UNIQUE / PRIMARY KEY / DEFAULT)"""
import pytest
import tempfile
import shutil
import os
from datetime import datetime, timezone

from apexbase import ApexClient


@pytest.fixture
def db():
    """Create a temporary ApexClient instance for each test."""
    tmp = tempfile.mkdtemp()
    client = ApexClient(tmp)
    yield client
    client.close()
    shutil.rmtree(tmp, ignore_errors=True)


class TestNotNullConstraint:
    """NOT NULL constraint enforcement on INSERT and UPDATE."""

    def test_not_null_reject_insert(self, db):
        db.execute("CREATE TABLE t1 (id INT NOT NULL, name TEXT)")
        db.execute("INSERT INTO t1 (id, name) VALUES (1, 'Alice')")
        with pytest.raises(Exception, match="NOT NULL"):
            db.execute("INSERT INTO t1 (id, name) VALUES (NULL, 'Bob')")

    def test_not_null_accept_valid(self, db):
        db.execute("CREATE TABLE t2 (id INT NOT NULL, name TEXT NOT NULL)")
        db.execute("INSERT INTO t2 (id, name) VALUES (1, 'Alice')")
        db.execute("INSERT INTO t2 (id, name) VALUES (2, 'Bob')")
        df = db.execute("SELECT COUNT(*) FROM t2").to_pandas()
        assert df.iloc[0, 0] == 2

    def test_not_null_reject_missing_column(self, db):
        db.execute("CREATE TABLE t3 (id INT NOT NULL, name TEXT NOT NULL)")
        with pytest.raises(Exception, match="NOT NULL"):
            db.execute("INSERT INTO t3 (id) VALUES (1)")

    def test_not_null_reject_update(self, db):
        db.execute("CREATE TABLE t4 (id INT NOT NULL, name TEXT NOT NULL)")
        db.execute("INSERT INTO t4 (id, name) VALUES (1, 'Alice')")
        with pytest.raises(Exception, match="NOT NULL"):
            db.execute("UPDATE t4 SET name = NULL WHERE id = 1")


class TestUniqueConstraint:
    """UNIQUE constraint enforcement on INSERT and UPDATE."""

    def test_unique_reject_duplicate_insert(self, db):
        db.execute("CREATE TABLE t5 (id INT, email TEXT UNIQUE)")
        db.execute("INSERT INTO t5 (id, email) VALUES (1, 'a@b.com')")
        with pytest.raises(Exception, match="UNIQUE"):
            db.execute("INSERT INTO t5 (id, email) VALUES (2, 'a@b.com')")

    def test_unique_allows_multiple_nulls(self, db):
        db.execute("CREATE TABLE t6 (id INT, email TEXT UNIQUE)")
        db.execute("INSERT INTO t6 (id, email) VALUES (1, NULL)")
        db.execute("INSERT INTO t6 (id, email) VALUES (2, NULL)")
        df = db.execute("SELECT COUNT(*) FROM t6").to_pandas()
        assert df.iloc[0, 0] == 2

    def test_unique_reject_batch_duplicates(self, db):
        db.execute("CREATE TABLE t7 (id INT, email TEXT UNIQUE)")
        with pytest.raises(Exception, match="UNIQUE"):
            db.execute("INSERT INTO t7 (id, email) VALUES (1, 'x@y.com'), (2, 'x@y.com')")

    def test_unique_reject_update(self, db):
        db.execute("CREATE TABLE t8 (id INT, email TEXT UNIQUE)")
        db.execute("INSERT INTO t8 (id, email) VALUES (1, 'a@b.com')")
        db.execute("INSERT INTO t8 (id, email) VALUES (2, 'c@d.com')")
        with pytest.raises(Exception, match="UNIQUE"):
            db.execute("UPDATE t8 SET email = 'a@b.com' WHERE id = 2")

    def test_unique_update_same_value_ok(self, db):
        """Updating a row to its own existing value should succeed."""
        db.execute("CREATE TABLE t9 (id INT, email TEXT UNIQUE)")
        db.execute("INSERT INTO t9 (id, email) VALUES (1, 'a@b.com')")
        # Updating to same value should not conflict with itself
        db.execute("UPDATE t9 SET email = 'a@b.com' WHERE id = 1")
        df = db.execute("SELECT email FROM t9 WHERE id = 1").to_pandas()
        assert df.iloc[0, 0] == "a@b.com"


class TestPrimaryKeyConstraint:
    """PRIMARY KEY constraint: implies NOT NULL + UNIQUE."""

    def test_pk_reject_duplicate(self, db):
        db.execute("CREATE TABLE t10 (uid INT PRIMARY KEY, name TEXT)")
        db.execute("INSERT INTO t10 (uid, name) VALUES (1, 'Alice')")
        with pytest.raises(Exception, match="PRIMARY KEY"):
            db.execute("INSERT INTO t10 (uid, name) VALUES (1, 'Bob')")

    def test_pk_implies_not_null(self, db):
        db.execute("CREATE TABLE t11 (uid INT PRIMARY KEY, name TEXT)")
        with pytest.raises(Exception, match="NOT NULL"):
            db.execute("INSERT INTO t11 (uid, name) VALUES (NULL, 'Alice')")

    def test_pk_multiple_rows_ok(self, db):
        db.execute("CREATE TABLE t12 (uid INT PRIMARY KEY, name TEXT)")
        db.execute("INSERT INTO t12 (uid, name) VALUES (1, 'Alice')")
        db.execute("INSERT INTO t12 (uid, name) VALUES (2, 'Bob')")
        db.execute("INSERT INTO t12 (uid, name) VALUES (3, 'Charlie')")
        df = db.execute("SELECT COUNT(*) FROM t12").to_pandas()
        assert df.iloc[0, 0] == 3


class TestDefaultConstraint:
    """DEFAULT value fill for missing columns on INSERT."""

    def test_default_int_fill(self, db):
        db.execute("CREATE TABLE t13 (id INT NOT NULL, score INT DEFAULT 100)")
        db.execute("INSERT INTO t13 (id) VALUES (1)")
        df = db.execute("SELECT score FROM t13 WHERE id = 1").to_pandas()
        assert df.iloc[0, 0] == 100

    def test_default_string_fill(self, db):
        db.execute("CREATE TABLE t14 (id INT NOT NULL, status TEXT DEFAULT 'active')")
        db.execute("INSERT INTO t14 (id) VALUES (1)")
        df = db.execute("SELECT status FROM t14 WHERE id = 1").to_pandas()
        assert df.iloc[0, 0] == "active"

    def test_default_not_null_with_default(self, db):
        """NOT NULL + DEFAULT should succeed when column is omitted."""
        db.execute("CREATE TABLE t15 (id INT NOT NULL, cnt INT NOT NULL DEFAULT 0)")
        db.execute("INSERT INTO t15 (id) VALUES (1)")
        df = db.execute("SELECT cnt FROM t15 WHERE id = 1").to_pandas()
        assert df.iloc[0, 0] == 0

    def test_default_override(self, db):
        """Explicit value should override DEFAULT."""
        db.execute("CREATE TABLE t16 (id INT NOT NULL, score INT DEFAULT 100)")
        db.execute("INSERT INTO t16 (id, score) VALUES (1, 42)")
        df = db.execute("SELECT score FROM t16 WHERE id = 1").to_pandas()
        assert df.iloc[0, 0] == 42

    def test_default_multiple_rows(self, db):
        """DEFAULT should fill for each row in a batch insert."""
        db.execute("CREATE TABLE t17 (id INT NOT NULL, score INT DEFAULT 50)")
        db.execute("INSERT INTO t17 (id) VALUES (1)")
        db.execute("INSERT INTO t17 (id) VALUES (2)")
        df = db.execute("SELECT SUM(score) FROM t17").to_pandas()
        assert df.iloc[0, 0] == 100

    def test_default_current_date_function(self, db):
        """CURRENT_DATE should be usable as a dynamic DEFAULT value."""
        before = datetime.now(timezone.utc).date().isoformat()
        db.execute("CREATE TABLE t18 (id INT NOT NULL, created TEXT DEFAULT CURRENT_DATE)")
        db.execute("INSERT INTO t18 (id) VALUES (1)")
        after = datetime.now(timezone.utc).date().isoformat()
        df = db.execute("SELECT created FROM t18 WHERE id = 1").to_pandas()
        assert df.iloc[0, 0] in {before, after}

    def test_default_unix_timestamp_function(self, db):
        """UNIX_TIMESTAMP() should be usable as a dynamic DEFAULT value."""
        before = int(datetime.now(timezone.utc).timestamp())
        db.execute("CREATE TABLE t19 (id INT NOT NULL, ts BIGINT DEFAULT UNIX_TIMESTAMP())")
        db.execute("INSERT INTO t19 (id) VALUES (1)")
        after = int(datetime.now(timezone.utc).timestamp())
        df = db.execute("SELECT ts FROM t19 WHERE id = 1").to_pandas()
        assert before <= df.iloc[0, 0] <= after

    def test_default_constant_expression_fill(self, db):
        """Row-independent constant expressions should be accepted in DEFAULT."""
        db.execute(
            "CREATE TABLE t20 (id INT NOT NULL, ttl INT DEFAULT (60 * 60), status TEXT DEFAULT LOWER('ACTIVE'))"
        )
        db.execute("INSERT INTO t20 (id) VALUES (1)")
        row = db.execute("SELECT ttl, status FROM t20 WHERE id = 1").to_dict()[0]
        assert row["ttl"] == 3600
        assert row["status"] == "active"

    def test_default_cast_expression_fill(self, db):
        """CAST constant expressions should be folded and stored as typed DEFAULT values."""
        db.execute(
            "CREATE TABLE t21 (id INT NOT NULL, created TEXT DEFAULT CAST('2026-01-02' AS DATE))"
        )
        db.execute("INSERT INTO t21 (id) VALUES (1)")
        row = db.execute("SELECT created FROM t21 WHERE id = 1").to_dict()[0]
        assert row["created"] == "2026-01-02"

    def test_insert_default_values(self, db):
        """INSERT DEFAULT VALUES should create one row populated from column defaults."""
        db.execute("CREATE TABLE t22 (id INT DEFAULT 7, status TEXT DEFAULT 'new')")
        db.execute("INSERT INTO t22 DEFAULT VALUES")
        row = db.execute("SELECT id, status FROM t22").to_dict()[0]
        assert row["id"] == 7
        assert row["status"] == "new"

    def test_insert_values_default_keyword(self, db):
        """VALUES(DEFAULT, ...) should use the target column's declared default."""
        db.execute(
            "CREATE TABLE t23 (id INT NOT NULL, score INT DEFAULT 50, status TEXT DEFAULT 'ready')"
        )
        db.execute("INSERT INTO t23 (id, score, status) VALUES (1, DEFAULT, DEFAULT)")
        row = db.execute("SELECT score, status FROM t23 WHERE id = 1").to_dict()[0]
        assert row["score"] == 50
        assert row["status"] == "ready"

    def test_default_expression_rejects_column_reference(self, db):
        """DEFAULT expressions must be row-independent."""
        with pytest.raises(Exception, match="cannot reference column"):
            db.execute("CREATE TABLE t_bad_default (a INT, b INT DEFAULT a + 1)")


class TestConstraintPersistence:
    """Constraints should survive save/reopen cycles."""

    def test_constraints_persist_after_reopen(self):
        tmp = tempfile.mkdtemp()
        try:
            # Create table with constraints
            db1 = ApexClient(tmp)
            db1.execute("CREATE TABLE tp1 (id INT PRIMARY KEY, name TEXT NOT NULL)")
            db1.execute("INSERT INTO tp1 (id, name) VALUES (1, 'Alice')")
            db1.close()

            # Reopen and verify constraints are still enforced
            db2 = ApexClient(tmp)
            db2.use_table("tp1")
            with pytest.raises(Exception, match="PRIMARY KEY"):
                db2.execute("INSERT INTO tp1 (id, name) VALUES (1, 'Bob')")
            with pytest.raises(Exception, match="NOT NULL"):
                db2.execute("INSERT INTO tp1 (id, name) VALUES (2, NULL)")
            db2.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
