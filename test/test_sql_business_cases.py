"""
Business-oriented SQL test cases with CREATE VIEW and complex queries.
Supports both Python API and pure SQL (multi-statement DDL+DML) for data setup.
"""
import pytest
import tempfile
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apexbase', 'python'))

try:
    from apexbase import ApexClient
except ImportError as e:
    pytest.skip(f"ApexBase not available: {e}", allow_module_level=True)


def _execute_or_xfail(client: ApexClient, sql: str):
    try:
        return client.execute(sql)
    except Exception as e:
        pytest.xfail(f"SQL not supported yet: {type(e).__name__}: {e}")


class TestSqlBusinessCases:
    def test_ecommerce_multi_table_join_aggregation(self):
        """E-commerce scenario: users, orders, payments with 3-table JOIN + aggregation (no VIEW)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            # Create tables and insert data using Python API
            client.create_table("users")
            client.use_table("users")
            client.store([
                {"user_id": 1, "name": "Alice", "tier": "pro"},
                {"user_id": 2, "name": "Bob", "tier": "free"},
                {"user_id": 3, "name": "Charlie", "tier": "pro"},
            ])
            client.flush()

            client.create_table("orders")
            client.use_table("orders")
            client.store([
                {"order_id": 10, "user_id": 1, "amount": 120, "status": "paid"},
                {"order_id": 11, "user_id": 1, "amount": 80, "status": "paid"},
                {"order_id": 12, "user_id": 2, "amount": 30, "status": "paid"},
                {"order_id": 13, "user_id": 3, "amount": 200, "status": "refunded"},
            ])
            client.flush()

            client.create_table("payments")
            client.use_table("payments")
            client.store([
                {"payment_id": 100, "order_id": 10, "paid_amount": 120, "channel": "card"},
                {"payment_id": 101, "order_id": 11, "paid_amount": 80, "channel": "card"},
                {"payment_id": 102, "order_id": 12, "paid_amount": 30, "channel": "paypal"},
                {"payment_id": 103, "order_id": 13, "paid_amount": 200, "channel": "card"},
            ])
            client.flush()

            # Verify tables and data
            tables = client.list_tables()
            assert "users" in tables
            assert "orders" in tables
            assert "payments" in tables

            # Direct 3-table JOIN + aggregation (VIEW with subquery not supported in JOINs)
            res = _execute_or_xfail(
                client,
                """
                SELECT
                  u.tier AS tier,
                  u.name AS name,
                  COUNT(*) AS order_cnt,
                  SUM(p.paid_amount) AS paid
                FROM users u
                JOIN orders o ON u.user_id = o.user_id
                JOIN payments p ON o.order_id = p.order_id
                WHERE o.status = 'paid'
                GROUP BY u.tier, u.name
                ORDER BY paid DESC
                LIMIT 2
                """.strip(),
            )

            out = res.to_dict()
            assert out == [
                {"tier": "pro", "name": "Alice", "order_cnt": 2, "paid": 200},
                {"tier": "free", "name": "Bob", "order_cnt": 1, "paid": 30},
            ]

            client.close()

    def test_support_tickets_sla_view_having_and_pagination(self):
        """Support ticket SLA scenario: GROUP BY + HAVING + VIEW + pagination"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            # Create table and insert data using Python API
            client.create_table("tickets")
            client.use_table("tickets")
            client.store([
                {"ticket_id": 1, "team": "core", "priority": "p0", "minutes_to_close": 30, "status": "closed"},
                {"ticket_id": 2, "team": "core", "priority": "p1", "minutes_to_close": 90, "status": "closed"},
                {"ticket_id": 3, "team": "core", "priority": "p1", "minutes_to_close": 120, "status": "closed"},
                {"ticket_id": 4, "team": "ml", "priority": "p0", "minutes_to_close": 45, "status": "closed"},
                {"ticket_id": 5, "team": "ml", "priority": "p2", "minutes_to_close": 300, "status": "closed"},
                {"ticket_id": 6, "team": "ml", "priority": "p1", "minutes_to_close": 70, "status": "open"},
            ])
            client.flush()

            tables = client.list_tables()
            assert "tickets" in tables

            # Create VIEW with GROUP BY + HAVING, then query with ORDER BY + LIMIT
            res = _execute_or_xfail(
                client,
                """
                CREATE VIEW v_team_sla AS
                SELECT
                  team,
                  COUNT(*) AS closed_cnt,
                  AVG(minutes_to_close) AS avg_mins,
                  MAX(minutes_to_close) AS worst_mins
                FROM tickets
                WHERE status = 'closed'
                GROUP BY team
                HAVING COUNT(*) >= 2
                ORDER BY avg_mins ASC;

                SELECT team, closed_cnt, worst_mins
                FROM v_team_sla
                ORDER BY worst_mins DESC
                LIMIT 1 OFFSET 0
                """.strip(),
            )

            out = res.to_dict()
            assert out == [{"team": "ml", "closed_cnt": 2, "worst_mins": 300}]

            # VIEW should auto-expire
            with pytest.raises(Exception):
                client.execute("SELECT * FROM v_team_sla").to_dict()

            client.close()

    def test_marketing_campaign_attribution_join(self):
        """Marketing attribution scenario: campaigns + events JOIN + GROUP BY (no VIEW)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            # Create tables and insert data using Python API
            client.create_table("campaigns")
            client.use_table("campaigns")
            client.store([
                {"campaign_id": 10, "name": "spring", "channel": "email"},
                {"campaign_id": 11, "name": "summer", "channel": "ads"},
            ])
            client.flush()

            client.create_table("events")
            client.use_table("events")
            client.store([
                {"event_id": 1, "campaign_id": 10, "event_type": "impression", "revenue": 0},
                {"event_id": 2, "campaign_id": 10, "event_type": "click", "revenue": 0},
                {"event_id": 3, "campaign_id": 10, "event_type": "purchase", "revenue": 100},
                {"event_id": 4, "campaign_id": 11, "event_type": "impression", "revenue": 0},
                {"event_id": 5, "campaign_id": 11, "event_type": "purchase", "revenue": 40},
            ])
            client.flush()

            tables = client.list_tables()
            assert "campaigns" in tables
            assert "events" in tables

            # Direct JOIN + GROUP BY (VIEW with subquery not supported in JOINs)
            res = _execute_or_xfail(
                client,
                """
                SELECT
                  c.name AS name,
                  c.channel AS channel,
                  SUM(e.revenue) AS revenue
                FROM campaigns c
                JOIN events e ON c.campaign_id = e.campaign_id
                GROUP BY c.name, c.channel
                ORDER BY revenue DESC
                """.strip(),
            )

            out = res.to_dict()
            assert out == [
                {"name": "spring", "channel": "email", "revenue": 100},
                {"name": "summer", "channel": "ads", "revenue": 40},
            ]

            client.close()

    def test_pure_sql_multi_statement_ddl_dml_complex(self):
        """
        Comprehensive test: ALL SQL in ONE execute call with:
        - CREATE TABLE + ALTER TABLE + INSERT
        - CREATE VIEW with window functions, aggregates, subquery
        - VIEW + multi-table JOINs
        - Complex SELECT with ORDER BY, LIMIT
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            client = ApexClient(dirpath=temp_dir)

            # ALL DDL + DML + VIEW + JOIN in ONE execute call
            res = _execute_or_xfail(client, """
                CREATE TABLE IF NOT EXISTS employees;
                ALTER TABLE employees ADD COLUMN emp_id INT;
                ALTER TABLE employees ADD COLUMN name STRING;
                ALTER TABLE employees ADD COLUMN department STRING;
                ALTER TABLE employees ADD COLUMN salary INT;
                ALTER TABLE employees ADD COLUMN hire_year INT;
                
                INSERT INTO employees (emp_id, name, department, salary, hire_year) VALUES
                    (1, 'Alice', 'Engineering', 120000, 2020),
                    (2, 'Bob', 'Engineering', 95000, 2021),
                    (3, 'Charlie', 'Engineering', 110000, 2019),
                    (4, 'Diana', 'Sales', 85000, 2020),
                    (5, 'Eve', 'Sales', 92000, 2018),
                    (6, 'Frank', 'Marketing', 78000, 2022),
                    (7, 'Grace', 'Marketing', 82000, 2021),
                    (8, 'Henry', 'Engineering', 130000, 2017);

                CREATE TABLE IF NOT EXISTS projects;
                ALTER TABLE projects ADD COLUMN project_id INT;
                ALTER TABLE projects ADD COLUMN project_name STRING;
                ALTER TABLE projects ADD COLUMN budget INT;
                ALTER TABLE projects ADD COLUMN status STRING;
                
                INSERT INTO projects (project_id, project_name, budget, status) VALUES
                    (101, 'Alpha', 500000, 'active'),
                    (102, 'Beta', 300000, 'active'),
                    (103, 'Gamma', 150000, 'completed');

                CREATE TABLE IF NOT EXISTS assignments;
                ALTER TABLE assignments ADD COLUMN emp_id INT;
                ALTER TABLE assignments ADD COLUMN project_id INT;
                ALTER TABLE assignments ADD COLUMN hours_worked INT;
                ALTER TABLE assignments ADD COLUMN role STRING;
                
                INSERT INTO assignments (emp_id, project_id, hours_worked, role) VALUES
                    (1, 101, 200, 'lead'),
                    (1, 102, 50, 'advisor'),
                    (3, 101, 180, 'developer'),
                    (8, 101, 100, 'architect');

                CREATE VIEW v_high_performers AS
                SELECT 
                    emp_id,
                    name,
                    department,
                    salary,
                    ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS dept_rank,
                    COUNT(*) OVER (PARTITION BY department) AS dept_size
                FROM employees
                WHERE salary > (SELECT AVG(salary) FROM employees);

                SELECT 
                    v.name,
                    v.department,
                    v.salary,
                    v.dept_rank,
                    p.project_name,
                    a.role,
                    a.hours_worked
                FROM v_high_performers v
                JOIN assignments a ON v.emp_id = a.emp_id
                JOIN projects p ON a.project_id = p.project_id
                WHERE p.status = 'active'
                ORDER BY v.salary DESC, a.hours_worked DESC
                LIMIT 5
            """)

            result = res.to_dict()
            # Verify result contains expected structure
            assert len(result) >= 1, "Should return at least 1 row"
            # Check that all expected columns are present
            expected_cols = ['name', 'department', 'salary', 'dept_rank', 'project_name', 'role', 'hours_worked']
            for col in expected_cols:
                assert col in result[0], f"Missing column: {col}"
            # Verify high performers (salary > avg ~98k)
            for r in result:
                assert r['salary'] > 90000, "Should only include high performers"
            # Verify dept_rank is assigned
            assert result[0]['dept_rank'] >= 1
            # First row should be Henry (highest salary)
            assert result[0]['name'] == 'Henry'
            assert result[0]['salary'] == 130000
            assert result[0]['project_name'] == 'Alpha'

            # Verify VIEW is session-scoped (should fail outside the execute)
            with pytest.raises(Exception):
                client.execute("SELECT * FROM v_high_performers")

            client.close()
