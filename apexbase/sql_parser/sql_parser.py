from dataclasses import dataclass
from typing import List, Union, Optional, Any
from enum import Enum, auto


class SQLSyntaxError(SyntaxError):
    """Raised when a SQL syntax error is encountered."""
    def __init__(self, message, line=None, column=None):
        super().__init__(message)
        self.line = line
        self.column = column


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
        # location可以保存 (行, 列) 或 (start_pos, end_pos)，此处沿用原字段含义
        self.location = location


class Literal(Node):
    """Literal node"""
    def __init__(self, value: Any, location: Optional[tuple] = None):
        super().__init__(NodeType.LITERAL, location)
        self.value = value
        # 以下 type 信息仅示例保留
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
    """Binary operator node, e.g. a > b, or a BETWEEN x AND y"""
    def __init__(
        self, left: Node, right: Union[Node, tuple], operator: str,
        location: Optional[tuple] = None
    ):
        super().__init__(NodeType.BINARY_OP, location)
        self.left = left
        self.right = right
        self.operator = operator
        # Validate BETWEEN operator parameters
        if operator.upper() == 'BETWEEN':
            if not isinstance(right, tuple) or len(right) != 2:
                raise ValueError("BETWEEN operator requires exactly two values")


class FunctionCall(Node):
    """Function call node, e.g. JSON_EXTRACT(x, '$.y')"""
    def __init__(
        self, name: str, args: List[Node],
        distinct: bool = False, location: Optional[tuple] = None
    ):
        super().__init__(NodeType.FUNCTION_CALL, location)
        self.name = name
        self.args = args
        self.distinct = distinct


class LogicalOp(Node):
    """Logical operator node, e.g. expr1 AND expr2"""
    def __init__(
        self, left: Node, operator: str, right: Optional[Node] = None,
        location: Optional[tuple] = None
    ):
        super().__init__(NodeType.LOGICAL_OP, location)
        self.left = left
        self.operator = operator
        self.right = right


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
    type: Union[str, TokenType]
    value: Any
    position: Optional[tuple] = None  # (start_pos, end_pos) 或 (line, col)

    def __str__(self):
        return f"Token({self.type}, {self.value})"

    def __repr__(self):
        return self.__str__()


class SQLLexer:
    """SQL lexer"""
    def __init__(self):
        # 在这里把 'REGEXP' 也加入操作符集合
        self.operators = {
            '=', '>', '<', '>=', '<=', '!=', '<>', 'LIKE', 'REGEXP'
        }
        self.logical = {'AND', 'OR', 'NOT'}
        self.keywords = {
            'IS', 'NULL', 'IN', 'BETWEEN', 'AND', 'OR', 'NOT',
            'CASE', 'WHEN', 'THEN', 'ELSE', 'END',
            'ORDER', 'BY', 'ASC', 'DESC',
            # 'LIKE' 单独放在上方 operators 里
        }
        self.functions = {
            'JSON_EXTRACT', 'CAST', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX',
            'COALESCE', 'NULLIF', 'IFNULL', 'LENGTH', 'UPPER', 'LOWER',
            'CREATE_FTS_INDEX', 'DROP_FTS_INDEX', 'MATCH_BM25'
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
        if self.peek() == '\n':
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
        """Read number (int or float)"""
        start_pos = self.pos
        num_str = ''

        # Optional negative sign
        if self.peek() == '-':
            num_str = '-'
            self.advance()

        # Integer part
        while self.peek().isdigit():
            num_str += self.peek()
            self.advance()

        # Decimal point
        if self.peek() == '.':
            num_str += '.'
            self.advance()
            while self.peek().isdigit():
                num_str += self.peek()
                self.advance()

        # Scientific notation
        if self.peek().lower() == 'e':
            num_str += 'e'
            self.advance()
            if self.peek() in '+-':
                num_str += self.peek()
                self.advance()
            while self.peek().isdigit():
                num_str += self.peek()
                self.advance()

        # Convert to float if '.' or 'e' in it
        if '.' in num_str or 'e' in num_str.lower():
            value = float(num_str)
        else:
            value = int(num_str)

        return Token(TokenType.NUMBER, value, (self.line, self.column))

    def read_string(self) -> Token:
        """Read string literal, supporting single or double quotes"""
        start_line, start_col = self.line, self.column
        quote_char = self.peek()
        self.advance()  # skip quote

        string_val = ''
        while True:
            c = self.peek()
            if c == '':
                raise SQLSyntaxError(
                    "Unterminated string literal", line=self.line, column=self.column
                )
            if c == quote_char:
                # end
                self.advance()
                return Token(TokenType.STRING, string_val, (start_line, start_col))
            else:
                string_val += c
                self.advance()

    def read_identifier(self) -> Token:
        """
        读取标识符、关键字、函数名或操作符单词。
        如果单词在 self.operators 集里（LIKE、REGEXP等），
        则返回 TokenType.OPERATOR。
        """
        start_line, start_col = self.line, self.column
        identifier = ''

        # Handle quoted identifier with backticks or double-quotes
        if self.peek() in ('"', '`'):
            quote = self.peek()
            self.advance()
            while self.pos < len(self.text) and self.peek() != quote:
                identifier += self.peek()
                self.advance()
            if self.pos >= len(self.text):
                raise SQLSyntaxError(
                    "Unterminated quoted identifier",
                    line=self.line, column=self.column
                )
            self.advance()  # skip end quote
            return Token(TokenType.IDENTIFIER, identifier, (start_line, start_col))

        # Normal identifier
        while self.peek().isalnum() or self.peek() == '_':
            identifier += self.peek()
            self.advance()

        upper_ident = identifier.upper()
        # 若在函数集合
        if upper_ident in self.functions:
            return Token(TokenType.FUNCTION, upper_ident, (start_line, start_col))

        # 若在操作符集合 => 生成 OPERATOR
        elif upper_ident in self.operators:
            return Token(TokenType.OPERATOR, upper_ident, (start_line, start_col))

        # 若在关键字
        elif upper_ident in self.keywords:
            token_type = getattr(TokenType, upper_ident, TokenType.KEYWORD)
            return Token(token_type, upper_ident, (start_line, start_col))

        else:
            # default as IDENTIFIER
            return Token(TokenType.IDENTIFIER, identifier, (start_line, start_col))

    def read_operator(self) -> Token:
        """
        读取符号类操作符(如 =, !=, >=, <>)。'REGEXP'是一种文本操作符，
        这里不处理，由 read_identifier() 处理。
        """
        start_line, start_col = self.line, self.column
        two_chars = self.text[self.pos : self.pos+2]
        if two_chars in ('>=', '<=', '!=', '<>'):
            op_val = two_chars
            self.advance()
            self.advance()
            return Token(TokenType.OPERATOR, op_val, (start_line, start_col))
        else:
            op_char = self.peek()
            self.advance()
            return Token(TokenType.OPERATOR, op_char, (start_line, start_col))

    def tokenize(self, text: str) -> List[Token]:
        self.reset(text)
        tokens = []
        while self.pos < len(self.text):
            self.skip_whitespace()
            if self.pos >= len(self.text):
                break

            ch = self.peek()

            # number?
            if ch.isdigit() or (ch == '-' and self.pos + 1 < len(self.text) and self.text[self.pos+1].isdigit()):
                tokens.append(self.read_number())
                continue

            # string? (single/double/backtick)
            if ch in ("'", '"', '`'):
                tokens.append(self.read_string())
                continue

            # identifier / keyword / function / (REGEXP / LIKE) operator
            if ch.isalpha() or ch == '_':
                tokens.append(self.read_identifier())
                continue

            # operator chars: =, <, >, !
            if ch in ('=', '>', '<', '!'):
                tokens.append(self.read_operator())
                continue

            # parentheses / commas / dots
            start_line, start_col = self.line, self.column
            if ch == '(':
                tokens.append(Token(TokenType.LPAREN, ch, (start_line, start_col)))
                self.advance()
                continue
            if ch == ')':
                tokens.append(Token(TokenType.RPAREN, ch, (start_line, start_col)))
                self.advance()
                continue
            if ch == ',':
                tokens.append(Token(TokenType.COMMA, ch, (start_line, start_col)))
                self.advance()
                continue
            if ch == '.':
                tokens.append(Token(TokenType.DOT, ch, (start_line, start_col)))
                self.advance()
                continue

            raise SQLSyntaxError(
                f"Invalid character '{ch}'",
                line=self.line,
                column=self.column
            )

        tokens.append(Token(TokenType.EOF, '', (self.line, self.column)))
        return tokens


class SQLParser:
    """SQL parser that builds an AST for a simplified subset of SQL expressions."""

    def __init__(self):
        self.tokens: List[Token] = []
        self.token_index: int = -1
        self.current_token: Optional[Token] = None

    def parse(self, text: str) -> Node:
        """Parse an input SQL expression and return the AST root."""
        lexer = SQLLexer()
        self.tokens = lexer.tokenize(text)
        self.token_index = 0
        self.current_token = self.tokens[0] if self.tokens else None

        # 解析一个 WHERE 级别的表达式
        ast_root = self.expr()

        # 若还有剩余 token 且不是 EOF => 语法有问题
        if self.current_token and self.current_token.type != TokenType.EOF:
            t = self.current_token
            line, col = (t.position if t.position else (None, None))
            raise SQLSyntaxError(
                f"Unexpected token after expression: {t.value}",
                line=line, column=col
            )

        return ast_root

    def advance(self):
        """Consume current token and move to next."""
        self.token_index += 1
        if self.token_index < len(self.tokens):
            self.current_token = self.tokens[self.token_index]
        else:
            self.current_token = None

    def eat(self, expected_type: TokenType):
        """Check if current token matches expected_type, otherwise raise syntax error."""
        if self.current_token and self.current_token.type == expected_type:
            tok = self.current_token
            self.advance()
            return tok
        else:
            ct = self.current_token
            line, col = (ct.position if ct and ct.position else (None, None))
            raise SQLSyntaxError(
                f"Expected token {expected_type}, got {ct}",
                line=line, column=col
            )

    def expr(self) -> Node:
        """
        expr := comparison ( (AND|OR) comparison )*
        """
        node = self.comparison()

        while self.current_token and self.current_token.type in (TokenType.AND, TokenType.OR):
            op_token = self.current_token
            self.advance()  # consume AND/OR
            right_node = self.comparison()
            node = LogicalOp(left=node, operator=op_token.value, right=right_node,
                             location=op_token.position)
        return node

    def comparison(self) -> Node:
        """
        comparison := term ( (OPERATOR|BETWEEN|IS) term )*
        e.g. a = b, a REGEXP 'x', a IS NULL, a BETWEEN b AND c, ...
        """
        node = self.term()
        while self.current_token and self.current_token.type in (
            TokenType.OPERATOR, TokenType.BETWEEN, TokenType.IS
        ):
            op_token = self.current_token
            self.advance()  # consume the operator/BETWEEN/IS

            if op_token.type == TokenType.BETWEEN:
                # parse BETWEEN x AND y
                start_val = self.term()
                if not self.current_token or self.current_token.type != TokenType.AND:
                    line, col = (op_token.position if op_token.position else (None, None))
                    raise SQLSyntaxError("Expected AND after BETWEEN start value",
                                         line=line, column=col)
                self.advance()  # consume AND
                end_val = self.term()
                node = BinaryOp(left=node, right=(start_val, end_val),
                                operator="BETWEEN", location=op_token.position)

            elif op_token.type == TokenType.IS:
                # parse IS [NOT] NULL
                if self.current_token and self.current_token.type == TokenType.NOT:
                    not_token = self.current_token
                    self.advance()  # consume NOT
                    if not self.current_token or self.current_token.type != TokenType.NULL:
                        line, col = (not_token.position if not_token.position else (None, None))
                        raise SQLSyntaxError("Expected NULL after IS NOT",
                                             line=line, column=col)
                    self.advance()  # consume NULL
                    node = BinaryOp(left=node, right=None, operator="IS NOT NULL",
                                    location=op_token.position)
                else:
                    # expect NULL
                    if not self.current_token or self.current_token.type != TokenType.NULL:
                        line, col = (op_token.position if op_token.position else (None, None))
                        raise SQLSyntaxError("Expected NULL after IS",
                                             line=line, column=col)
                    self.advance()  # consume NULL
                    node = BinaryOp(left=node, right=None, operator="IS NULL",
                                    location=op_token.position)
            else:
                # normal operator: =, <, >, <=, >=, LIKE, REGEXP, etc.
                right_node = self.term()
                node = BinaryOp(left=node, right=right_node,
                                operator=op_token.value, location=op_token.position)
        return node

    def term(self) -> Node:
        """term := factor"""
        return self.factor()

    def factor(self) -> Node:
        """
        factor := IDENTIFIER | NUMBER | STRING | FUNCTION(...) | ( expr )
        并在此处检查：若 IDENTIFIER 后面紧跟另一个 IDENTIFIER/NUMBER/STRING，不符合语法 => 抛错
        """
        token = self.current_token
        if not token:
            raise SQLSyntaxError("Unexpected end of input in factor", line=None, column=None)

        # IDENTIFIER
        if token.type == TokenType.IDENTIFIER:
            ident_token = token
            self.advance()  # consume IDENTIFIER
            ident_node = Identifier(name=ident_token.value, location=ident_token.position)

            # **关键：检查下一个 token**，若也是 IDENTIFIER/NUMBER/STRING，通常表示语法错误
            if self.current_token and self.current_token.type in (
                TokenType.IDENTIFIER, TokenType.NUMBER, TokenType.STRING
            ):
                next_t = self.current_token
                line, col = (next_t.position if next_t.position else (None, None))
                raise SQLSyntaxError(
                    f"Syntax error: unexpected token '{next_t.value}' "
                    f"after identifier '{ident_token.value}'",
                    line=line, column=col
                )
            return ident_node

        # NUMBER
        elif token.type == TokenType.NUMBER:
            self.advance()
            return Literal(value=token.value, location=token.position)

        # STRING
        elif token.type == TokenType.STRING:
            self.advance()
            return Literal(value=token.value, location=token.position)

        # FUNCTION
        elif token.type == TokenType.FUNCTION:
            return self.function_call()

        # (
        elif token.type == TokenType.LPAREN:
            lp_token = token
            self.advance()  # consume '('
            subexpr = self.expr()
            if not self.current_token or self.current_token.type != TokenType.RPAREN:
                line, col = (self.current_token.position if self.current_token else (None, None))
                raise SQLSyntaxError("Missing closing parenthesis", line=line, column=col)
            self.advance()  # consume ')'
            return subexpr

        # If none of the above
        line_col = token.position
        raise SQLSyntaxError(
            f"Unexpected token in factor: {token.value}",
            line=line_col[0] if line_col else None,
            column=line_col[1] if line_col else None
        )

    def function_call(self) -> Node:
        """Parse function call: FUNCTION '(' expr (, expr)* ')'"""
        func_token = self.current_token
        if not func_token:
            raise SQLSyntaxError("Unexpected end in function call", line=None, column=None)

        self.advance()  # consume the FUNCTION token
        lp = self.eat(TokenType.LPAREN)  # must be '('

        args = []
        while self.current_token and self.current_token.type != TokenType.RPAREN:
            args.append(self.expr())
            if self.current_token and self.current_token.type == TokenType.COMMA:
                self.advance()  # consume comma

        # finally must see ')'
        rp = self.eat(TokenType.RPAREN)
        return FunctionCall(name=func_token.value, args=args, location=func_token.position)
