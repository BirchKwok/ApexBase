from .dialect import SQLDialect
from .sql_parser import SQLParser
from .sql_generator import SQLGenerator

def parse_sql_to_ast(sql_text: str):
    """Convenience function to parse SQL text into an AST."""
    parser = SQLParser()
    ast = parser.parse(sql_text)
    return ast

def generate_sql(ast_node, dialect: SQLDialect = SQLDialect.SQLITE):
    """Generate SQL text (and parameters) from an AST, for a given dialect."""
    gen = SQLGenerator(dialect)
    sql_str = gen.generate(ast_node)
    params = gen.get_parameters()
    return sql_str, params
