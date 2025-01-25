from dataclasses import dataclass
from typing import List, Union, Optional, Any
import re
from enum import Enum, auto

class NodeType(Enum):
    """节点类型枚举"""
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
    """AST节点基类"""
    def __init__(self, node_type: NodeType, location: Optional[tuple] = None):
        self.type = node_type
        self.location = location  # (start_pos, end_pos)

class Literal(Node):
    """字面量节点"""
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
    """标识符节点"""
    def __init__(self, name: str, quoted: bool = False, location: Optional[tuple] = None):
        super().__init__(NodeType.IDENTIFIER, location)
        self.name = name
        self.quoted = quoted

class JsonPath(Node):
    """JSON路径节点"""
    def __init__(self, field: str, path: str, location: Optional[tuple] = None):
        super().__init__(NodeType.JSON_PATH, location)
        self.field = field
        self.path = path

class BinaryOp(Node):
    """二元操作符节点"""
    def __init__(self, left: Node, right: Union[Node, tuple], operator: str, location: Optional[tuple] = None):
        super().__init__(NodeType.BINARY_OP, location)
        self.left = left
        self.right = right
        self.operator = operator
        
        # 验证BETWEEN操作符的参数
        if operator == 'BETWEEN':
            if not isinstance(right, tuple) or len(right) != 2:
                raise ValueError("BETWEEN operator requires exactly two values")

class FunctionCall(Node):
    """函数调用节点"""
    def __init__(self, name: str, args: List[Node], distinct: bool = False, location: Optional[tuple] = None):
        super().__init__(NodeType.FUNCTION_CALL, location)
        self.name = name
        self.args = args
        self.distinct = distinct

class LogicalOp(Node):
    """逻辑操作符节点"""
    def __init__(self, left: Node, operator: str, right: Optional[Node] = None, location: Optional[tuple] = None):
        super().__init__(NodeType.LOGICAL_OP, location)
        self.left = left
        self.operator = operator
        self.right = right

class OrderBy(Node):
    """排序节点"""
    def __init__(self, expressions: List[tuple[Node, str]], location: Optional[tuple] = None):
        super().__init__(NodeType.ORDER_BY, location)
        self.expressions = expressions

class Between(Node):
    """BETWEEN节点"""
    def __init__(self, expr: Node, start: Node, end: Node, location: Optional[tuple] = None):
        super().__init__(NodeType.BETWEEN, location)
        self.expr = expr
        self.start = start
        self.end = end

class InList(Node):
    """IN列表节点"""
    def __init__(self, expr: Node, values: List[Node], location: Optional[tuple] = None):
        super().__init__(NodeType.IN_LIST, location)
        self.expr = expr
        self.values = values

class IsNull(Node):
    """IS NULL节点"""
    def __init__(self, expr: Node, is_not: bool = False, location: Optional[tuple] = None):
        super().__init__(NodeType.IS_NULL, location)
        self.expr = expr
        self.is_not = is_not

class CaseWhen(Node):
    """CASE WHEN节点"""
    def __init__(self, conditions: List[tuple[Node, Node]], else_result: Optional[Node] = None, location: Optional[tuple] = None):
        super().__init__(NodeType.CASE_WHEN, location)
        self.conditions = conditions
        self.else_result = else_result

class TokenType(Enum):
    """Token 类型枚举"""
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
    """词法单元"""
    type: str
    value: Any
    position: Optional[int] = None

    def __str__(self):
        return f"Token({self.type}, {self.value})"

    def __repr__(self):
        return self.__str__()

class SQLLexer:
    """SQL 词法分析器"""
    def __init__(self):
        # 所有操作符和关键字都使用大写形式存储
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
        """重置词法分析器状态"""
        self.text = text.strip()
        self.pos = 0
        self.line = 1
        self.column = 1

    def peek(self) -> str:
        """查看下一个字符但不移动位置"""
        if self.pos < len(self.text):
            return self.text[self.pos]
        return ''

    def advance(self):
        """移动到下一个字符"""
        if self.text[self.pos] == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        self.pos += 1

    def skip_whitespace(self):
        """跳过空白字符"""
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.advance()

    def read_number(self) -> Token:
        """读取数字"""
        start_pos = self.pos
        num = ''
        
        # 处理负号
        if self.peek() == '-':
            num = '-'
            self.advance()
        
        # 读取整数部分
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            num += self.text[self.pos]
            self.advance()
        
        # 处理小数点
        if self.pos < len(self.text) and self.text[self.pos] == '.':
            num += '.'
            self.advance()
            # 读取小数部分
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                num += self.text[self.pos]
                self.advance()
        
        # 处理科学计数法
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
        """读取字符串字面量"""
        start_pos = self.pos
        quote = self.text[self.pos]
        self.advance()  # 跳过开始引号
        string = ''

        while self.pos < len(self.text):
            if self.text[self.pos] == quote and (self.pos + 1 >= len(self.text) or self.text[self.pos + 1] != quote):
                # 字符串结束
                self.advance()  # 跳过结束引号
                return Token(TokenType.STRING, string, (start_pos, self.pos))
            elif self.text[self.pos] == quote and self.pos + 1 < len(self.text) and self.text[self.pos + 1] == quote:
                # 处理转义的引号
                string += quote
                self.advance()  # 跳过第一个引号
                self.advance()  # 跳过第二个引号
            else:
                string += self.text[self.pos]
                self.advance()

        # 如果到达文本末尾但没有找到结束引号
        raise ValueError(f"Unterminated string at position {start_pos}")

    def read_identifier(self) -> Token:
        """读取标识符或关键字"""
        start_pos = self.pos
        identifier = ''
        
        # 处理引号包裹的标识符
        if self.text[self.pos] in ('"', '`'):
            quote = self.text[self.pos]
            self.advance()  # 跳过开始引号
            while self.pos < len(self.text) and self.text[self.pos] != quote:
                identifier += self.text[self.pos]
                self.advance()
            if self.pos >= len(self.text):
                raise ValueError(f"Unterminated quoted identifier at position {start_pos}")
            self.advance()  # 跳过结束引号
            return Token(TokenType.IDENTIFIER, identifier, (start_pos, self.pos))
        
        # 处理普通标识符
        while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] == '_'):
            identifier += self.text[self.pos]
            self.advance()
        
        # 转换为大写进行比较，但保留原始大小写
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
        """读取操作符"""
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
        """将输入文本转换为token列表"""
        self.reset(text)
        tokens = []
        
        while self.pos < len(self.text):
            char = self.text[self.pos]
            
            # 跳过空白字符
            if char.isspace():
                self.skip_whitespace()
                continue
            
            # 处理数字
            if char.isdigit() or (char == '-' and self.pos + 1 < len(self.text) and self.text[self.pos + 1].isdigit()):
                tokens.append(self.read_number())
                continue
            
            # 处理字符串
            if char in ('"', "'", '`'):
                tokens.append(self.read_string())
                continue
            
            # 处理标识符和关键字
            if char.isalpha() or char == '_' or char in ('"', '`'):
                tokens.append(self.read_identifier())
                continue
            
            # 处理操作符
            if char in {'=', '>', '<', '!'}:
                tokens.append(self.read_operator())
                continue
            
            # 处理括号和其他符号
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
    """SQL 解析器"""
    def __init__(self):
        self.tokens = []
        self.token_index = -1

    def parse(self, text):
        """解析 SQL 查询文本"""
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
        检查当前token是否为预期类型，如果是则消耗它
        
        Parameters:
            token_type: Union[str, TokenType]
                预期的token类型
        """
        if isinstance(token_type, str):
            token_type = getattr(TokenType, token_type)
            
        if self.current_token and self.current_token.type == token_type:
            token = self.current_token
            self.advance()
            return token
        raise ValueError(f"Expected {token_type}, got {self.current_token}")

    def expr(self):
        """解析表达式"""
        node = self.comparison()

        while self.current_token and self.current_token.type in {TokenType.AND, TokenType.OR}:
            token = self.current_token
            self.advance()
            node = LogicalOp(node, token.value, self.comparison())

        return node

    def comparison(self):
        """解析比较表达式"""
        node = self.term()

        while self.current_token and self.current_token.type in {TokenType.OPERATOR, TokenType.BETWEEN, TokenType.IS}:
            token = self.current_token
            if token.type == TokenType.BETWEEN:
                self.advance()
                start_value = self.term()
                if not self.current_token or self.current_token.type != TokenType.AND:
                    raise ValueError("Expected AND after BETWEEN start value")
                self.advance()  # 跳过 AND
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
        """解析项"""
        return self.factor()

    def factor(self):
        """解析因子"""
        token = self.current_token
        
        if not token:
            raise ValueError("Unexpected end of input")
        
        if token.type == TokenType.IDENTIFIER:
            node = Identifier(token.value)
            self.advance()
            
            # 检查是否有比较运算符
            if self.current_token and self.current_token.type == TokenType.OPERATOR:
                operator = self.current_token.value
                self.advance()
                
                # 获取右侧值
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
        """解析函数调用"""
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
    """SQL 生成器"""
    def __init__(self):
        self.parameters = []

    def reset(self):
        """重置生成器状态"""
        self.parameters = []

    def get_parameters(self):
        """获取参数列表"""
        return self.parameters

    def generate(self, node):
        """生成 SQL 表达式"""
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
                
                # 第一个参数是字段名
                field = self.generate(node.args[0])
                
                # 第二个参数是JSON路径
                if not isinstance(node.args[1], Literal) or not isinstance(node.args[1].value, str):
                    raise ValueError("Second argument of json_extract must be a string literal")
                
                self.parameters.append(node.args[1].value)
                return f"json_extract({field}, ?)"
            else:
                args = [self.generate(arg) for arg in node.args]
                return f"{node.name}({', '.join(args)})"
        else:
            raise ValueError(f"Unsupported node type: {type(node)}") 
            raise ValueError(f"Unsupported node type: {type(node)}") 