from typing import Union, Any
from .dialect import SQLDialect
from .sql_parser import (
    Node, BinaryOp, LogicalOp, Literal, Identifier, FunctionCall, JsonPath
)

class SQLGenerator:
    """
    Generate SQL from the AST, supporting two dialects: SQLite, DuckDB.
    We handle expressions, placeholders, function calls, etc.
    """

    def __init__(self, dialect: SQLDialect = SQLDialect.SQLITE):
        self.dialect = dialect
        self.parameters = []
        self.param_index = 0

    def reset(self):
        self.parameters.clear()
        self.param_index = 0

    def get_parameters(self):
        return self.parameters

    def generate(self, node: Node) -> str:
        """
        Dispatch method. For each node type, we have a visitor function:
        visit_binaryop, visit_logicalop, visit_literal, ...
        """
        method_name = f"visit_{node.__class__.__name__.lower()}"
        method = getattr(self, method_name, None)
        if not method:
            raise ValueError(f"No visitor method for node type {node.__class__.__name__}")
        return method(node)

    def visit_binaryop(self, node: BinaryOp) -> str:
        left_sql = self.generate(node.left)
        op = node.operator.upper()

        if op == "BETWEEN":
            # node.right => (start_node, end_node)
            start_node, end_node = node.right
            start_sql = self._value_or_placeholder(start_node)
            end_sql   = self._value_or_placeholder(end_node)
            return f"{left_sql} BETWEEN {start_sql} AND {end_sql}"

        if op == "IS NULL":
            return f"{left_sql} IS NULL"
        if op == "IS NOT NULL":
            return f"{left_sql} IS NOT NULL"

        # e.g. =, <, >, >=, <=, <>, !=, LIKE ...
        right_sql = self._value_or_placeholder(node.right)
        return f"{left_sql} {node.operator} {right_sql}"

    def visit_logicalop(self, node: LogicalOp) -> str:
        # (expr1 AND expr2), (expr1 OR expr2)
        left_sql = self.generate(node.left)
        right_sql = self.generate(node.right)
        return f"({left_sql} {node.operator} {right_sql})"

    def visit_literal(self, node: Literal) -> str:
        # For a literal, we typically want to use placeholders to prevent SQL injection, etc.
        return self.add_parameter(node.value)

    def visit_identifier(self, node: Identifier) -> str:
        # Both SQLite & DuckDB generally accept double-quoted identifiers
        return f'"{node.name}"'

    def visit_functioncall(self, node: FunctionCall) -> str:
        func_name = node.name.upper()
        if func_name == "JSON_EXTRACT":
            if len(node.args) != 2:
                raise ValueError("JSON_EXTRACT requires exactly 2 arguments")
            # e.g. json_extract(field, path)
            field_sql = self.generate(node.args[0])
            path_sql  = self._value_or_placeholder(node.args[1])
            # both SQLite (with JSON1) & DuckDB have `json_extract` with similar usage
            return f"json_extract({field_sql}, {path_sql})"
        # else normal function
        arg_sqls = [self.generate(arg) for arg in node.args]
        return f"{node.name}({', '.join(arg_sqls)})"

    def visit_jsonpath(self, node: JsonPath) -> str:
        """
        If you had a separate node for JSON path, handle it here.
        Example: `json_extract(field, "path")`
        """
        # In a real design, you'd decide how to represent "field" as an Identifier or string.
        return f'json_extract({node.field}, "{node.path}")'

    # -------------- Helper Methods -------------------

    def add_parameter(self, value: Any) -> str:
        """Add a parameter to the list, return the placeholder according to dialect."""
        self.parameters.append(value)
        self.param_index += 1

        if self.dialect == SQLDialect.SQLITE:
            # e.g. ?1, ?2, ...
            return f"?{self.param_index}"
        elif self.dialect == SQLDialect.DUCKDB:
            # Usually DuckDB is okay with '?'
            return "?"
        else:
            # fallback
            return "?"

    def _value_or_placeholder(self, node: Union[Node, None]) -> str:
        """Generate SQL or use placeholder if it's a literal."""
        if isinstance(node, Literal):
            return self.add_parameter(node.value)
        else:
            return self.generate(node)
