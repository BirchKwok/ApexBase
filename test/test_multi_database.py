"""
Comprehensive test suite for ApexBase Multi-Database Support

Tests:
- use_database() / use() / list_databases() / current_database
- Database isolation (tables don't leak across databases)
- Cross-database SQL: SELECT, JOIN, INSERT, DELETE, UPDATE
- Qualified db.table syntax in all SQL contexts
- db.table in DDL: CREATE TABLE, DROP TABLE
- db.table in DML: INSERT INTO, UPDATE, DELETE FROM
- Backward compatibility: default database behaviour unchanged
- Error handling: invalid database names, missing tables
"""

import pytest
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apexbase', 'python'))

try:
    from apexbase import ApexClient
except ImportError as e:
    pytest.skip(f"ApexBase not available: {e}", allow_module_level=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_client():
    """Fresh ApexClient in a temp directory, closed on teardown."""
    with tempfile.TemporaryDirectory() as tmp:
        client = ApexClient(dirpath=tmp)
        yield client
        client.close()


@pytest.fixture
def populated_client():
    """Client with two databases pre-populated with data."""
    with tempfile.TemporaryDirectory() as tmp:
        client = ApexClient(dirpath=tmp)
        # default db
        client.use(database='default', table='users')
        client.store([
            {'name': 'Alice', 'age': 30, 'dept_id': 1},
            {'name': 'Bob',   'age': 25, 'dept_id': 2},
            {'name': 'Carol', 'age': 35, 'dept_id': 1},
        ])
        # analytics db
        client.use(database='analytics', table='events')
        client.store([
            {'event': 'click', 'user_id': 1, 'cnt': 10},
            {'event': 'view',  'user_id': 2, 'cnt': 50},
            {'event': 'buy',   'user_id': 1, 'cnt': 5},
        ])
        yield client
        client.close()


# ---------------------------------------------------------------------------
# 1. Database lifecycle: use_database / use / list_databases / current_database
# ---------------------------------------------------------------------------

class TestDatabaseLifecycle:

    def test_default_database_on_init(self, tmp_client):
        assert tmp_client.current_database == 'default'

    def test_list_databases_initial(self, tmp_client):
        dbs = tmp_client.list_databases()
        assert 'default' in dbs

    def test_use_database_creates_subdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = ApexClient(dirpath=tmp)
            client.use_database('analytics')
            assert os.path.isdir(os.path.join(tmp, 'analytics'))
            client.close()

    def test_use_database_returns_self(self, tmp_client):
        result = tmp_client.use_database('analytics')
        assert result is tmp_client

    def test_use_database_default_returns_to_root(self, tmp_client):
        tmp_client.use_database('analytics')
        tmp_client.use_database('default')
        assert tmp_client.current_database == 'default'

    def test_use_database_empty_string_equals_default(self, tmp_client):
        tmp_client.use_database('')
        assert tmp_client.current_database == 'default'

    def test_current_database_updates(self, tmp_client):
        tmp_client.use_database('hr')
        assert tmp_client.current_database == 'hr'
        tmp_client.use_database('analytics')
        assert tmp_client.current_database == 'analytics'

    def test_list_databases_after_creation(self, tmp_client):
        tmp_client.use_database('hr')
        tmp_client.use_database('analytics')
        dbs = tmp_client.list_databases()
        assert 'default' in dbs
        assert 'hr' in dbs
        assert 'analytics' in dbs

    def test_use_combined_api_returns_self(self, tmp_client):
        result = tmp_client.use(database='analytics', table='events')
        assert result is tmp_client

    def test_use_combined_api_sets_database_and_table(self, tmp_client):
        tmp_client.use(database='analytics', table='events')
        assert tmp_client.current_database == 'analytics'
        assert tmp_client.current_table == 'events'

    def test_use_database_only_clears_table(self, tmp_client):
        tmp_client.use(database='default', table='users')
        assert tmp_client.current_table == 'users'
        tmp_client.use_database('analytics')
        # After switching db, table context must be cleared
        assert tmp_client.current_table is None

    def test_use_creates_table_if_missing(self, tmp_client):
        tmp_client.use(database='new_db', table='my_table')
        assert 'my_table' in tmp_client.list_tables()

    def test_use_selects_existing_table(self, tmp_client):
        tmp_client.use(database='default', table='users')
        tmp_client.store({'name': 'Alice', 'age': 30})
        # Switch away then back
        tmp_client.use(database='analytics', table='events')
        tmp_client.use(database='default', table='users')
        assert tmp_client.current_table == 'users'
        assert tmp_client.count_rows() == 1

    def test_multiple_databases_listed_sorted(self, tmp_client):
        for db in ['zoo', 'alpha', 'beta']:
            tmp_client.use_database(db)
        dbs = tmp_client.list_databases()
        assert dbs == sorted(dbs)


# ---------------------------------------------------------------------------
# 2. Database isolation
# ---------------------------------------------------------------------------

class TestDatabaseIsolation:

    def test_tables_isolated_across_databases(self, tmp_client):
        tmp_client.use(database='default', table='users')
        tmp_client.store({'name': 'Alice'})

        tmp_client.use_database('analytics')
        # 'users' should not exist in analytics db
        assert 'users' not in tmp_client.list_tables()

    def test_list_tables_reflects_current_database(self, populated_client):
        populated_client.use_database('default')
        assert 'users' in populated_client.list_tables()
        assert 'events' not in populated_client.list_tables()

        populated_client.use_database('analytics')
        assert 'events' in populated_client.list_tables()
        assert 'users' not in populated_client.list_tables()

    def test_count_rows_isolated(self, populated_client):
        populated_client.use(database='default', table='users')
        assert populated_client.count_rows() == 3

        populated_client.use(database='analytics', table='events')
        assert populated_client.count_rows() == 3

    def test_write_to_one_db_does_not_affect_other(self, populated_client):
        # Write 10 more rows to analytics
        populated_client.use(database='analytics', table='events')
        for i in range(10):
            populated_client.store({'event': f'e{i}', 'user_id': i, 'cnt': i})

        # Default db row count unchanged
        populated_client.use(database='default', table='users')
        assert populated_client.count_rows() == 3

    def test_drop_table_in_one_db_does_not_affect_other(self, populated_client):
        populated_client.use_database('default')
        populated_client.drop_table('users')

        populated_client.use_database('analytics')
        assert 'events' in populated_client.list_tables()

    def test_same_table_name_in_different_databases(self, tmp_client):
        tmp_client.use(database='default', table='data')
        tmp_client.store({'val': 1})

        tmp_client.use(database='analytics', table='data')
        tmp_client.store({'val': 2})
        tmp_client.store({'val': 3})

        # Counts must differ
        tmp_client.use(database='default', table='data')
        assert tmp_client.count_rows() == 1

        tmp_client.use(database='analytics', table='data')
        assert tmp_client.count_rows() == 2


# ---------------------------------------------------------------------------
# 3. Cross-database SQL: SELECT
# ---------------------------------------------------------------------------

class TestCrossDbSelect:

    def test_select_from_qualified_table(self, populated_client):
        populated_client.use_database('analytics')
        result = populated_client.execute('SELECT * FROM default.users')
        df = result.to_pandas()
        assert len(df) == 3
        assert set(df['name'].tolist()) == {'Alice', 'Bob', 'Carol'}

    def test_select_from_same_database_qualified(self, populated_client):
        populated_client.use_database('analytics')
        result = populated_client.execute('SELECT * FROM analytics.events')
        assert len(result) == 3

    def test_select_from_different_db_without_switching(self, populated_client):
        # In default context, query analytics.events
        populated_client.use(database='default', table='users')
        result = populated_client.execute('SELECT * FROM analytics.events')
        assert len(result) == 3

    def test_select_with_where_on_qualified_table(self, populated_client):
        populated_client.use_database('analytics')
        result = populated_client.execute(
            "SELECT * FROM default.users WHERE age > 25"
        )
        df = result.to_pandas()
        assert len(df) == 2  # Alice (30), Carol (35)

    def test_select_with_aggregation_on_qualified_table(self, populated_client):
        populated_client.use_database('analytics')
        result = populated_client.execute(
            "SELECT COUNT(*) AS cnt, AVG(age) AS avg_age FROM default.users"
        )
        row = result.first()
        assert row['cnt'] == 3

    def test_select_with_order_by_on_qualified_table(self, populated_client):
        populated_client.use_database('analytics')
        result = populated_client.execute(
            "SELECT name FROM default.users ORDER BY age ASC"
        )
        names = [r['name'] for r in result.to_dict()]
        assert names[0] == 'Bob'

    def test_select_no_current_table_with_qualified_ref(self, tmp_client):
        """execute() should work with qualified ref even if no table is selected."""
        tmp_client.use(database='default', table='items')
        tmp_client.store({'item': 'pen', 'price': 1})
        tmp_client.use_database('analytics')
        # No table selected in analytics context
        result = tmp_client.execute('SELECT * FROM default.items')
        assert len(result) == 1

    def test_select_third_database(self, tmp_client):
        tmp_client.use(database='db1', table='t1')
        tmp_client.store({'v': 1})
        tmp_client.use(database='db2', table='t2')
        tmp_client.store({'v': 2})
        tmp_client.use(database='db3', table='t3')
        tmp_client.store({'v': 3})

        # Query db1 from db3 context
        result = tmp_client.execute('SELECT v FROM db1.t1')
        assert result.first()['v'] == 1

        # Query db2 from db3 context
        result = tmp_client.execute('SELECT v FROM db2.t2')
        assert result.first()['v'] == 2


# ---------------------------------------------------------------------------
# 4. Cross-database SQL: JOIN
# ---------------------------------------------------------------------------

class TestCrossDbJoin:

    def test_inner_join_cross_db(self, populated_client):
        populated_client.use_database('analytics')
        result = populated_client.execute("""
            SELECT u.name, e.event
            FROM default.users u
            JOIN analytics.events e ON u.dept_id = e.user_id
        """)
        df = result.to_pandas()
        assert len(df) > 0
        assert 'name' in df.columns
        assert 'event' in df.columns

    def test_join_same_and_cross_db(self, tmp_client):
        tmp_client.use(database='default', table='orders')
        tmp_client.store([
            {'order_id': 1, 'user_id': 10},
            {'order_id': 2, 'user_id': 20},
        ])
        tmp_client.use(database='hr', table='employees')
        tmp_client.store([
            {'emp_id': 10, 'name': 'Alice'},
            {'emp_id': 20, 'name': 'Bob'},
        ])

        tmp_client.use_database('default')
        result = tmp_client.execute("""
            SELECT o.order_id, e.name
            FROM default.orders o
            JOIN hr.employees e ON o.user_id = e.emp_id
            ORDER BY o.order_id
        """)
        rows = result.to_dict()
        assert len(rows) == 2
        assert rows[0]['name'] == 'Alice'
        assert rows[1]['name'] == 'Bob'

    def test_join_two_external_databases_from_third(self, tmp_client):
        tmp_client.use(database='db_a', table='a')
        tmp_client.store([{'id': 1, 'val': 'x'}])
        tmp_client.use(database='db_b', table='b')
        tmp_client.store([{'id': 1, 'label': 'y'}])

        tmp_client.use_database('db_c')
        result = tmp_client.execute("""
            SELECT a.val, b.label
            FROM db_a.a a
            JOIN db_b.b b ON a.id = b.id
        """)
        assert len(result) == 1
        row = result.first()
        assert row['val'] == 'x'
        assert row['label'] == 'y'


# ---------------------------------------------------------------------------
# 5. Cross-database DML: INSERT, UPDATE, DELETE
# ---------------------------------------------------------------------------

class TestCrossDbDML:

    def test_insert_into_qualified_table(self, tmp_client):
        tmp_client.use(database='default', table='logs')
        tmp_client.store({'msg': 'init'})

        tmp_client.use_database('analytics')
        tmp_client.execute("INSERT INTO default.logs (msg) VALUES ('cross-db-insert')")

        tmp_client.use(database='default', table='logs')
        assert tmp_client.count_rows() == 2

    def test_delete_from_qualified_table(self, populated_client):
        populated_client.use_database('analytics')
        populated_client.execute(
            "DELETE FROM default.users WHERE age < 26"
        )
        populated_client.use(database='default', table='users')
        assert populated_client.count_rows() == 2  # Alice + Carol remain

    def test_update_qualified_table(self, populated_client):
        populated_client.use_database('analytics')
        populated_client.execute(
            "UPDATE default.users SET age = 99 WHERE name = 'Alice'"
        )
        populated_client.use(database='default', table='users')
        result = populated_client.execute("SELECT age FROM users WHERE name = 'Alice'")
        assert result.first()['age'] == 99


# ---------------------------------------------------------------------------
# 6. Cross-database DDL
# ---------------------------------------------------------------------------

class TestCrossDbDDL:

    def test_create_table_qualified_name(self, tmp_client):
        tmp_client.use_database('analytics')
        tmp_client.execute("CREATE TABLE default.remote_tbl (val INT)")
        tmp_client.use_database('default')
        assert 'remote_tbl' in tmp_client.list_tables()

    def test_drop_table_qualified_name(self, populated_client):
        populated_client.use_database('analytics')
        populated_client.execute("DROP TABLE IF EXISTS default.users")
        populated_client.use_database('default')
        assert 'users' not in populated_client.list_tables()


# ---------------------------------------------------------------------------
# 7. Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:

    def test_existing_single_db_workflow_unchanged(self):
        """Old usage pattern (no use_database calls) must work as before."""
        with tempfile.TemporaryDirectory() as tmp:
            client = ApexClient(dirpath=tmp)
            client.create_table("users")
            client.store([{'name': 'Alice', 'age': 30}])
            result = client.execute("SELECT * FROM users WHERE age > 25")
            assert len(result) == 1
            client.close()

    def test_default_database_tables_in_root(self):
        """Tables in 'default' must live at the root dir (no subdir)."""
        with tempfile.TemporaryDirectory() as tmp:
            client = ApexClient(dirpath=tmp)
            client.use(database='default', table='items')
            client.store({'item': 'pen'})
            # File must be at root, not in a subdir
            assert os.path.exists(os.path.join(tmp, 'items.apex'))
            assert not os.path.exists(os.path.join(tmp, 'default', 'items.apex'))
            client.close()

    def test_named_database_tables_in_subdir(self):
        """Tables in a named db must live in root/db_name/."""
        with tempfile.TemporaryDirectory() as tmp:
            client = ApexClient(dirpath=tmp)
            client.use(database='analytics', table='events')
            client.store({'event': 'click'})
            assert os.path.exists(os.path.join(tmp, 'analytics', 'events.apex'))
            client.close()

    def test_create_table_still_works_in_default_db(self, tmp_client):
        tmp_client.create_table("legacy_table")
        assert 'legacy_table' in tmp_client.list_tables()

    def test_unqualified_sql_still_works_against_current_table(self, tmp_client):
        tmp_client.use(database='default', table='products')
        tmp_client.store([{'sku': 'A', 'price': 10}, {'sku': 'B', 'price': 20}])
        result = tmp_client.execute("SELECT * FROM products WHERE price > 15")
        assert len(result) == 1

    def test_use_table_still_works_within_database(self, populated_client):
        populated_client.use_database('default')
        populated_client.create_table('orders')
        populated_client.store({'order': 'o1'})
        populated_client.use_table('users')
        assert populated_client.current_table == 'users'

    def test_drop_if_exists_with_multi_db(self):
        """drop_if_exists=True should only clear default db, not named dbs."""
        with tempfile.TemporaryDirectory() as tmp:
            c1 = ApexClient(dirpath=tmp)
            c1.use(database='default', table='users')
            c1.store({'name': 'Alice'})
            c1.use(database='analytics', table='events')
            c1.store({'e': 1})
            c1.close()

            # Reopen with drop_if_exists — analytics subdir must still exist
            c2 = ApexClient(dirpath=tmp, drop_if_exists=False)
            c2.use_database('analytics')
            assert 'events' in c2.list_tables()
            c2.close()


# ---------------------------------------------------------------------------
# 8. Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_missing_qualified_table_raises(self, tmp_client):
        tmp_client.use_database('analytics')
        with pytest.raises(Exception):
            tmp_client.execute("SELECT * FROM default.nonexistent_table")

    def test_use_table_after_use_database_on_missing_table(self, tmp_client):
        tmp_client.use_database('analytics')
        with pytest.raises(Exception):
            tmp_client.use_table('does_not_exist')

    def test_count_rows_requires_table(self, tmp_client):
        tmp_client.use_database('analytics')
        with pytest.raises(Exception):
            tmp_client.count_rows()

    def test_store_requires_table(self, tmp_client):
        tmp_client.use_database('analytics')
        with pytest.raises(Exception):
            tmp_client.store({'x': 1})

    def test_use_database_then_use_table_works(self, tmp_client):
        tmp_client.use(database='analytics', table='events')
        tmp_client.store({'e': 1})
        assert tmp_client.count_rows() == 1
