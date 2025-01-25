from dataclasses import dataclass
from typing import List, Union, Optional, Any
import re
from enum import Enum, auto

class NodeType(Enum):
    """Node type enumeration"""
    LITERAL = auto()
    IDENTIFIER = auto()
    BINARY_OP = auto()
    LOGICAL_OP = auto()
    FUNCTION_CALL = auto()
    JSON_PATH = auto()
    ORDER_BY = auto()
    GROUP_BY = auto()
    HAVING = auto()
    LIMIT = auto()
    OFFSET = auto()
    BETWEEN = auto()
    IN_LIST = auto()
    IS_NULL = auto()
    CASE_WHEN = auto()
    SUBQUERY = auto()

class Node:
    """AST node base class"""
    def __init__(self, node_type: NodeType, location: Optional[tuple] = None):
        self.type = node_type
        self.location = location  # (start_pos, end_pos)

class Literal(Node):
    """Literal node"""
    def __init__(self, value: Any, location: Optional[tuple] = None):
        super().__init__(NodeType.LITERAL, location)
        self.value = value
        if isinstance(value, bool):
            self.value_type = 'boolean'
        elif isinstance(value, (int, float)):
            self.value_type = 'number'
        elif value is None:
            self.value_type = 'null'
        else:
            self.value_type = 'string'

class Identifier(Node):
    """Identifier node"""
    def __init__(self, name: str, quoted: bool = False, location: Optional[tuple] = None):
        super().__init__(NodeType.IDENTIFIER, location)
        self.name = name
        self.quoted = quoted

class JsonPath(Node):
    """JSON path node"""
    def __init__(self, field: str, path: str, location: Optional[tuple] = None):
        super().__init__(NodeType.JSON_PATH, location)
        self.field = field
        self.path = path

class BinaryOp(Node):
    """Binary operator node"""
    def __init__(self, left: Node, right: Union[Node, tuple], operator: str, location: Optional[tuple] = None):
        super().__init__(NodeType.BINARY_OP, location)
        self.left = left
        self.right = right
        self.operator = operator
        
        # Validate BETWEEN operator parameters
        if operator == 'BETWEEN':
            if not isinstance(right, tuple) or len(right) != 2:
                raise ValueError("BETWEEN operator requires exactly two values")

class FunctionCall(Node):
    """Function call node"""
    def __init__(self, name: str, args: List[Node], distinct: bool = False, location: Optional[tuple] = None):
        super().__init__(NodeType.FUNCTION_CALL, location)
        self.name = name
        self.args = args
        self.distinct = distinct

class LogicalOp(Node):
    """Logical operator node"""
    def __init__(self, left: Node, operator: str, right: Optional[Node] = None, location: Optional[tuple] = None):
        super().__init__(NodeType.LOGICAL_OP, location)
        self.left = left
        self.operator = operator
        self.right = right

class OrderBy(Node):
    """Order by node"""
    def __init__(self, expressions: List[tuple[Node, str]], location: Optional[tuple] = None):
        super().__init__(NodeType.ORDER_BY, location)
        self.expressions = expressions

class Between(Node):
    """Between node"""
    def __init__(self, expr: Node, start: Node, end: Node, location: Optional[tuple] = None):
        super().__init__(NodeType.BETWEEN, location)
        self.expr = expr
        self.start = start
        self.end = end

class InList(Node):
    """In list node"""
    def __init__(self, expr: Node, values: List[Node], location: Optional[tuple] = None):
        super().__init__(NodeType.IN_LIST, location)
        self.expr = expr
        self.values = values

class IsNull(Node):
    """IS NULL node"""
    def __init__(self, expr: Node, is_not: bool = False, location: Optional[tuple] = None):
        super().__init__(NodeType.IS_NULL, location)
        self.expr = expr
        self.is_not = is_not

class CaseWhen(Node):
    """CASE WHEN node"""
    def __init__(self, conditions: List[tuple[Node, Node]], else_result: Optional[Node] = None, location: Optional[tuple] = None):
        super().__init__(NodeType.CASE_WHEN, location)
        self.conditions = conditions
        self.else_result = else_result

class TokenType(Enum):
    """Token type enumeration"""
    IDENTIFIER = 'IDENTIFIER'
    NUMBER = 'NUMBER'
    STRING = 'STRING'
    OPERATOR = 'OPERATOR'
    LOGICAL = 'LOGICAL'
    LPAREN = 'LPAREN'
    RPAREN = 'RPAREN'
    COMMA = 'COMMA'
    DOT = 'DOT'
    FUNCTION = 'FUNCTION'
    KEYWORD = 'KEYWORD'
    EOF = 'EOF'
    BETWEEN = 'BETWEEN'
    AND = 'AND'
    OR = 'OR'
    NOT = 'NOT'
    IN = 'IN'
    IS = 'IS'
    NULL = 'NULL'
    LIKE = 'LIKE'
    CASE = 'CASE'
    WHEN = 'WHEN'
    THEN = 'THEN'
    ELSE = 'ELSE'
    END = 'END'
    ORDER = 'ORDER'
    BY = 'BY'
    ASC = 'ASC'
    DESC = 'DESC'

@dataclass
class Token:
    """Lexical unit"""
    type: str
    value: Any
    position: Optional[int] = None

    def __str__(self):
        return f"Token({self.type}, {self.value})"

    def __repr__(self):
        return self.__str__()

class SQLLexer:
    """SQL lexer"""
    def __init__(self):
        # All operators and keywords are stored in uppercase
        self.operators = {'=', '>', '<', '>=', '<=', '!=', 'LIKE', 'IN', 'IS NULL', 'IS NOT NULL', 'BETWEEN'}
        self.logical = {'AND', 'OR', 'NOT'}
        self.keywords = {
            'IS', 'NULL', 'NOT', 'LIKE', 'IN', 'BETWEEN', 'AND', 'OR',
            'CASE', 'WHEN', 'THEN', 'ELSE', 'END',
            'ORDER', 'BY', 'ASC', 'DESC'
        }
        self.functions = {
            'JSON_EXTRACT', 'CAST', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX',
            'COALESCE', 'NULLIF', 'IFNULL', 'LENGTH', 'UPPER', 'LOWER'
        }
        self.text = ''
        self.pos = 0
        self.line = 1
        self.column = 1

    def reset(self, text: str):
        """Reset lexer state"""
        self.text = text.strip()
        self.pos = 0
        self.line = 1
        self.column = 1

    def peek(self) -> str:
        """Peek at the next character without moving position"""
        if self.pos < len(self.text):
            return self.text[self.pos]
        return ''

    def advance(self):
        """Move to the next character"""
        if self.text[self.pos] == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        self.pos += 1

    def skip_whitespace(self):
        """Skip whitespace characters"""
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.advance()

    def read_number(self) -> Token:
        """Read number"""
        start_pos = self.pos
        num = ''
        
        # Handle negative sign
        if self.peek() == '-':
            num = '-'
            self.advance()
        
        # Read integer part
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            num += self.text[self.pos]
            self.advance()
        
        # Handle decimal point
        if self.pos < len(self.text) and self.text[self.pos] == '.':
            num += '.'
            self.advance()
            # Read fractional part
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                num += self.text[self.pos]
                self.advance()
        
        # Handle scientific notation
        if self.pos < len(self.text) and self.text[self.pos].lower() == 'e':
            num += self.text[self.pos]
            self.advance()
            if self.pos < len(self.text) and self.text[self.pos] in '+-':
                num += self.text[self.pos]
                self.advance()
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                num += self.text[self.pos]
                self.advance()
        
        return Token(TokenType.NUMBER, float(num) if '.' in num or 'e' in num.lower() else int(num), (start_pos, self.pos))

    def read_string(self) -> Token:
        """Read string literal"""
        start_pos = self.pos
        quote = self.text[self.pos]
        self.advance()  # Skip start quote
        string = ''

        while self.pos < len(self.text):
            if self.text[self.pos] == quote and (self.pos + 1 >= len(self.text) or self.text[self.pos + 1] != quote):
                # String ends
                self.advance()  # Skip end quote
                return Token(TokenType.STRING, string, (start_pos, self.pos))
            elif self.text[self.pos] == quote and self.pos + 1 < len(self.text) and self.text[self.pos + 1] == quote:
                # Handle escaped quotes
                string += quote
                self.advance()  # Skip first quote
                self.advance()  # Skip second quote
            else:
                string += self.text[self.pos]
                self.advance()

        # If end of text is reached without finding a closing quote
        raise ValueError(f"Unterminated string at position {start_pos}")

    def read_identifier(self) -> Token:
        """Read identifier or keyword"""
        start_pos = self.pos
        identifier = ''
        
        # Handle quoted identifiers
        if self.text[self.pos] in ('"', '`'):
            quote = self.text[self.pos]
            self.advance()  # Skip start quote
            while self.pos < len(self.text) and self.text[self.pos] != quote:
                identifier += self.text[self.pos]
                self.advance()
            if self.pos >= len(self.text):
                raise ValueError(f"Unterminated quoted identifier at position {start_pos}")
            self.advance()  # Skip end quote
            return Token(TokenType.IDENTIFIER, identifier, (start_pos, self.pos))
        
        # Handle normal identifiers
        while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] == '_'):
            identifier += self.text[self.pos]
            self.advance()
        
        # Convert to uppercase for comparison, but preserve original case
        upper_identifier = identifier.upper()
        if upper_identifier in self.functions:
            return Token(TokenType.FUNCTION, upper_identifier, (start_pos, self.pos))
        elif upper_identifier in self.keywords:
            if upper_identifier == 'LIKE':
                return Token(TokenType.OPERATOR, upper_identifier, (start_pos, self.pos))
            token_type = getattr(TokenType, upper_identifier, TokenType.KEYWORD)
            return Token(token_type, upper_identifier, (start_pos, self.pos))
        else:
            return Token(TokenType.IDENTIFIER, identifier, (start_pos, self.pos))

    def read_operator(self) -> Token:
        """Read operator"""
        start_pos = self.pos
        if self.text[self.pos:self.pos + 2] in {'>=', '<=', '!=', '<>'}:
            op = self.text[self.pos:self.pos + 2]
            self.advance()
            self.advance()
            return Token(TokenType.OPERATOR, op, (start_pos, self.pos))
        else:
            op = self.text[self.pos]
            self.advance()
            return Token(TokenType.OPERATOR, op, (start_pos, self.pos))

    def tokenize(self, text: str) -> List[Token]:
        """Convert input text to token list"""
        self.reset(text)
        tokens = []
        
        while self.pos < len(self.text):
            char = self.text[self.pos]
            
            # Skip whitespace characters
            if char.isspace():
                self.skip_whitespace()
                continue
            
            # Handle numbers
            if char.isdigit() or (char == '-' and self.pos + 1 < len(self.text) and self.text[self.pos + 1].isdigit()):
                tokens.append(self.read_number())
                continue
            
            # Handle strings
            if char in ('"', "'", '`'):
                tokens.append(self.read_string())
                continue
            
            # Handle identifiers and keywords
            if char.isalpha() or char == '_' or char in ('"', '`'):
                tokens.append(self.read_identifier())
                continue
            
            # Handle operators
            if char in {'=', '>', '<', '!'}:
                tokens.append(self.read_operator())
                continue
            
            # Handle parentheses and other symbols
            if char == '(':
                tokens.append(Token(TokenType.LPAREN, char, (self.pos, self.pos + 1)))
                self.advance()
                continue
            
            if char == ')':
                tokens.append(Token(TokenType.RPAREN, char, (self.pos, self.pos + 1)))
                self.advance()
                continue
            
            if char == ',':
                tokens.append(Token(TokenType.COMMA, char, (self.pos, self.pos + 1)))
                self.advance()
                continue
            
            if char == '.':
                tokens.append(Token(TokenType.DOT, char, (self.pos, self.pos + 1)))
                self.advance()
                continue
            
            if char == '%':
                tokens.append(Token(TokenType.STRING, char, (self.pos, self.pos + 1)))
                self.advance()
                continue
            
            raise ValueError(f"Invalid character '{char}' at position {self.pos}")
        
        return tokens

class SQLParser:
    """SQL parser"""
    def __init__(self):
        self.tokens = []
        self.token_index = -1

    def parse(self, text):
        """Parse SQL query text"""
        lexer = SQLLexer()
        self.tokens = lexer.tokenize(text)
        self.token_index = 0
        self.current_token = self.tokens[0] if self.tokens else None
        return self.expr()

    def advance(self):
        self.token_index += 1
        if self.token_index < len(self.tokens):
            self.current_token = self.tokens[self.token_index]
        else:
            self.current_token = None

    def eat(self, token_type):
        """
        Check if the current token is the expected type, and if so, consume it
        
        Parameters:
            token_type: Union[str, TokenType]
                Expected token type
        """
        if isinstance(token_type, str):
            token_type = getattr(TokenType, token_type)
            
        if self.current_token and self.current_token.type == token_type:
            token = self.current_token
            self.advance()
            return token
        raise ValueError(f"Expected {token_type}, got {self.current_token}")

    def expr(self):
        """Parse expression"""
        node = self.comparison()

        while self.current_token and self.current_token.type in {TokenType.AND, TokenType.OR}:
            token = self.current_token
            self.advance()
            node = LogicalOp(node, token.value, self.comparison())

        return node

    def comparison(self):
        """Parse comparison expression"""
        node = self.term()

        while self.current_token and self.current_token.type in {TokenType.OPERATOR, TokenType.BETWEEN, TokenType.IS}:
            token = self.current_token
            if token.type == TokenType.BETWEEN:
                self.advance()
                start_value = self.term()
                if not self.current_token or self.current_token.type != TokenType.AND:
                    raise ValueError("Expected AND after BETWEEN start value")
                self.advance()  # Skip AND
                end_value = self.term()
                node = BinaryOp(node, (start_value, end_value), 'BETWEEN')
            elif token.type == TokenType.IS:
                self.advance()
                if self.current_token and self.current_token.type == TokenType.NOT:
                    self.advance()
                    if not self.current_token or self.current_token.type != TokenType.NULL:
                        raise ValueError("Expected NULL after IS NOT")
                    self.advance()
                    node = BinaryOp(node, None, 'IS NOT NULL')
                else:
                    if not self.current_token or self.current_token.type != TokenType.NULL:
                        raise ValueError("Expected NULL after IS")
                    self.advance()
                    node = BinaryOp(node, None, 'IS NULL')
            else:
                self.advance()
                node = BinaryOp(node, self.term(), token.value)

        return node

    def term(self):
        """Parse term"""
        return self.factor()

    def factor(self):
        """Parse factor"""
        token = self.current_token
        
        if not token:
            raise ValueError("Unexpected end of input")
        
        if token.type == TokenType.IDENTIFIER:
            node = Identifier(token.value)
            self.advance()
            
            # Check for comparison operator
            if self.current_token and self.current_token.type == TokenType.OPERATOR:
                operator = self.current_token.value
                self.advance()
                
                # Get right value
                if not self.current_token:
                    raise ValueError("Expected value after operator")
                
                if self.current_token.type == TokenType.NUMBER:
                    value = self.current_token.value
                    self.advance()
                    right = Literal(value, 'number')
                elif self.current_token.type == TokenType.STRING:
                    value = self.current_token.value
                    self.advance()
                    right = Literal(value, 'string')
                else:
                    right = self.factor()
                
                return BinaryOp(node, right, operator)
            
            return node
        
        elif token.type == TokenType.NUMBER:
            value = token.value
            self.advance()
            return Literal(value, 'number')
        
        elif token.type == TokenType.STRING:
            value = token.value
            self.advance()
            return Literal(value, 'string')
        
        elif token.type == TokenType.FUNCTION:
            return self.function_call()
        
        elif token.type == TokenType.LPAREN:
            self.advance()
            node = self.expr()
            self.eat(TokenType.RPAREN)
            return node
        
        raise ValueError(f"Unexpected token: {token}")

    def function_call(self):
        """Parse function call"""
        token = self.current_token
        self.advance()
        self.eat(TokenType.LPAREN)
        args = []

        while True:
            args.append(self.expr())
            if self.current_token.type != TokenType.COMMA:
                break
            self.advance()

        self.eat(TokenType.RPAREN)
        return FunctionCall(token.value, args)

class SQLGenerator:
    """SQL generator"""
    def __init__(self):
        self.parameters = []

    def reset(self):
        """Reset generator state"""
        self.parameters = []

    def get_parameters(self):
        """Get parameter list"""
        return self.parameters

    def generate(self, node):
        """Generate SQL expression"""
        if isinstance(node, BinaryOp):
            left = self.generate(node.left)
            right = self.generate(node.right)
            return f"{left} {node.operator} {right}"
        elif isinstance(node, LogicalOp):
            left = self.generate(node.left)
            right = self.generate(node.right)
            return f"{left} {node.operator} {right}"
        elif isinstance(node, Identifier):
            return f'"{node.name}"'
        elif isinstance(node, Literal):
            self.parameters.append(node.value)
            return "?"
        elif isinstance(node, FunctionCall):
            if node.name.lower() == 'json_extract':
                if len(node.args) != 2:
                    raise ValueError("json_extract function requires exactly 2 arguments")
                
                # First argument is field name
                field = self.generate(node.args[0])
                
                # Second argument is JSON path
                if not isinstance(node.args[1], Literal) or not isinstance(node.args[1].value, str):
                    raise ValueError("Second argument of json_extract must be a string literal")
                
                self.parameters.append(node.args[1].value)
                return f"json_extract({field}, ?)"
            else:
                args = [self.generate(arg) for arg in node.args]
                return f"{node.name}({', '.join(args)})"
        else:
            raise ValueError(f"Unsupported node type: {type(node)}") 
