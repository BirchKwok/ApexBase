from .sql_parser import *

class SQLGenerator:
    """SQL generator"""

    def __init__(self):
        self.parameters = []
        self.param_index = 0

    def get_parameters(self):
        return self.parameters

    def reset(self):
        self.parameters = []
        self.param_index = 0

    def add_parameter(self, value):
        self.parameters.append(value)
        return "?"

    def generate(self, node):
        method = f'visit_{node.__class__.__name__.lower()}'
        visitor = getattr(self, method, None)
        if visitor is None:
            raise ValueError(f"No visitor found for {node.__class__.__name__}")
        return visitor(node)

    def visit_binaryop(self, node):
        left = self.generate(node.left)
        if node.operator == 'BETWEEN':
            if not isinstance(node.right, tuple) or len(node.right) != 2:
                raise ValueError("BETWEEN operator requires two values")
            start_val = self.add_parameter(node.right[0].value)
            end_val = self.add_parameter(node.right[1].value)
            return f"{left} BETWEEN {start_val} AND {end_val}"
        elif node.operator == 'IS NOT NULL':
            return f"{left} IS NOT NULL"
        elif node.operator == 'IS NULL':
            return f"{left} IS NULL"
        right = self.generate(node.right)
        if isinstance(node.right, Literal):
            right = self.add_parameter(node.right.value)
        return f"{left} {node.operator} {right}"

    def visit_logicalop(self, node):
        left = self.generate(node.left)
        right = self.generate(node.right)
        return f"({left}) {node.operator} ({right})"

    def visit_literal(self, node):
        return str(node.value)

    def visit_identifier(self, node):
        return f'"{node.name}"'

    def visit_jsonpath(self, node):
        return f'json_extract({node.field}, "{node.path}")'

    def visit_functioncall(self, node):
        if node.name.upper() == 'JSON_EXTRACT':
            if len(node.args) != 2:
                raise ValueError("JSON_EXTRACT function requires exactly 2 arguments")
            field = self.generate(node.args[0])
            path = node.args[1].value if isinstance(node.args[1], Literal) else node.args[1]
            return f'json_extract({field}, "{path}")'
        args = [self.generate(arg) for arg in node.args]
        return f'{node.name}({", ".join(args)})' 