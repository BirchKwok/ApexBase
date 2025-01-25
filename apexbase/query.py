from typing import List, Union, Dict, Any
import json
import re
from .sql_parser import SQLParser, SQLGenerator


class Query:
    """
    The FieldsQuery class is used to query data in the fields_cache.
    Supports direct SQL-like query syntax for filtering records.
    
    Examples:
        - Basic comparison: "age > 18"
        - Range query: "10 < days <= 300"
        - Text search: "name LIKE '%John%'"
        - Multiple conditions: "age > 18 AND city = 'New York'"
        - JSON field access: "json_extract(data, '$.address.city') = 'Beijing'"
        - Numeric operations: "CAST(json_extract(data, '$.price') AS REAL) * CAST(json_extract(data, '$.quantity') AS REAL) > 1000"
    """
    def __init__(self, storage):
        """
        Initialize the FieldsQuery class.

        Parameters:
            storage: Storage
                The storage object.
        """
        self.storage = storage
        self.parser = SQLParser()
        self.generator = SQLGenerator()
        # SQL语法组件
        self.sql_components = {
            'operators': {'=', '>', '<', '>=', '<=', '!=', 'LIKE', 'IN', 'IS', 'NOT', 'NULL', 'BETWEEN', 'AND', 'OR'},
            'logical': {'AND', 'OR', 'NOT'},
            'order': {'ORDER', 'BY', 'ASC', 'DESC'},
            'special': {'1=1'},  # 特殊条件
            'functions': {
                'CAST', 'AS', 'INTEGER', 'REAL', 'TEXT', 'JSON_EXTRACT', 'JSON_TYPE', 'JSON_VALID',
                'JSON_ARRAY', 'JSON_OBJECT', 'JSON_QUOTE', 'JSON_GROUP_ARRAY', 'JSON_GROUP_OBJECT',
                'JSON_ARRAY_LENGTH', 'JSON_REMOVE', 'JSON_REPLACE', 'JSON_SET', 'JSON_INSERT'
            }  # 函数和类型
        }

    def _quote_identifier(self, identifier: str) -> str:
        """
        正确转义 SQLite 标识符。

        Parameters:
            identifier: str
                需要转义的标识符

        Returns:
            str: 转义后的标识符
        """
        return f'"{identifier}"'

    def _parse_query(self, query_filter: str) -> Dict[str, str]:
        """
        解析查询字符串,提取 WHERE 和 ORDER BY 子句。

        Args:
            query_filter: 查询字符串

        Returns:
            Dict[str, str]: 包含 'where' 和 'order' 键的字典
        """
        if not query_filter:
            return {'where': '1=1', 'order': ''}

        # 分离 ORDER BY 子句
        parts = query_filter.split(' ORDER BY ', 1)
        where_clause = parts[0].strip()
        order_clause = f'ORDER BY {parts[1].strip()}' if len(parts) > 1 else ''

        # 如果 WHERE 子句为空,使用 1=1
        if not where_clause:
            where_clause = '1=1'

        return {
            'where': where_clause,
            'order': order_clause
        }

    def _validate_order_by(self, order_clause: str) -> bool:
        """
        验证ORDER BY子句。

        Parameters:
            order_clause: str
                ORDER BY子句

        Returns:
            bool: 是否有效
        """
        if not order_clause:
            return True

        parts = order_clause.upper().split()
        if len(parts) < 3:
            return False

        if parts[0] != 'ORDER' or parts[1] != 'BY':
            return False

        # 验证排序方向（如果指定）
        if len(parts) > 3 and parts[-1] not in {'ASC', 'DESC'}:
            return False

        return True

    def _validate_where(self, where_clause: str) -> bool:
        """
        验证WHERE子句。

        Parameters:
            where_clause: str
                WHERE子句

        Returns:
            bool: 是否有效

        Raises:
            ValueError: 当查询语法无效时抛出
        """
        if not where_clause or where_clause.upper() == '1=1':
            return True

        try:
            # 预处理：将表达式转换为标准形式
            # 1. 保护字符串字面量（包括单引号和双引号）
            def preserve_string(match):
                return f"STRING_LITERAL_{len(self._string_literals)}"
            
            self._string_literals = []
            cleaned = where_clause
            # 处理双引号字符串
            cleaned = re.sub(r'"([^"]*)"', lambda m: (self._string_literals.append(m.group(1)), preserve_string(m))[1], cleaned)
            # 处理单引号字符串
            cleaned = re.sub(r"'([^']*)'", lambda m: (self._string_literals.append(m.group(1)), preserve_string(m))[1], cleaned)
            
            # 2. 保护函数调用
            def preserve_function(match):
                func_name = match.group(1).upper()
                args = match.group(2)
                return f"{func_name}_CALL_{len(self._function_calls)}"
            
            self._function_calls = []
            cleaned = re.sub(r'(json_extract|cast)\s*\(([^)]+)\)', lambda m: (self._function_calls.append(m.group(0)), preserve_function(m))[1], cleaned, flags=re.IGNORECASE)
            
            # 3. 将其余部分转为大写（保护中文字符）
            def uppercase_non_chinese(match):
                text = match.group(0)
                return text if any('\u4e00' <= c <= '\u9fff' for c in text) else text.upper()
            
            cleaned = re.sub(r'\S+', uppercase_non_chinese, cleaned)
            
            # 4. 标准化空白字符
            cleaned = re.sub(r'\s+', ' ', cleaned)
            
            # 5. 分割成 tokens
            tokens = cleaned.split()
            if not tokens:
                raise ValueError("Invalid query syntax: empty WHERE clause")
            
            # 6. 验证每个 token
            i = 0
            while i < len(tokens):
                token = tokens[i]
                
                # 检查是否是字符串字面量
                if token.startswith('STRING_LITERAL_'):
                    i += 1
                    continue
                
                # 检查是否是函数调用
                if token.startswith(('JSON_EXTRACT_CALL_', 'CAST_CALL_')):
                    i += 1
                    if i < len(tokens):
                        if tokens[i] in {'=', '>', '<', '>=', '<=', '!='}:
                            i += 1
                            if i < len(tokens):
                                i += 1  # 跳过值
                            else:
                                raise ValueError(f"Invalid query syntax: missing value after operator in {where_clause}")
                        elif tokens[i].upper() == 'BETWEEN':
                            i += 1
                            if i < len(tokens):
                                i += 1  # 跳过第一个值
                                if i < len(tokens) and tokens[i].upper() == 'AND':
                                    i += 1
                                    if i < len(tokens):
                                        i += 1  # 跳过第二个值
                                    else:
                                        raise ValueError(f"Invalid query syntax: missing second value after BETWEEN in {where_clause}")
                                else:
                                    raise ValueError(f"Invalid query syntax: expected AND after BETWEEN value in {where_clause}")
                            else:
                                raise ValueError(f"Invalid query syntax: missing value after BETWEEN in {where_clause}")
                        elif tokens[i].upper() == 'IS':
                            i += 1
                            if i < len(tokens) and tokens[i].upper() == 'NOT':
                                i += 1
                            if i < len(tokens) and tokens[i].upper() == 'NULL':
                                i += 1
                            else:
                                raise ValueError(f"Invalid query syntax: expected NULL after IS [NOT] in {where_clause}")
                        else:
                            raise ValueError(f"Invalid query syntax: invalid operator after function call in {where_clause}")
                    else:
                        raise ValueError(f"Invalid query syntax: unexpected end after function call in {where_clause}")
                    continue
                
                # 检查是否是逻辑操作符
                if token.upper() in {'AND', 'OR'}:
                    i += 1
                    continue
                
                # 检查是否是字段名
                if re.match(r'^[A-Z_][A-Z0-9_]*$', token):
                    i += 1
                    if i < len(tokens) and (tokens[i] in {'=', '>', '<', '>=', '<=', '!='} or tokens[i].upper() in {'LIKE', 'IN', 'IS'}):
                        i += 1
                        if tokens[i-1].upper() == 'IS':
                            if i < len(tokens) and tokens[i].upper() == 'NOT':
                                i += 1
                            if i < len(tokens) and tokens[i].upper() == 'NULL':
                                i += 1
                            else:
                                raise ValueError(f"Invalid query syntax: expected NULL after IS [NOT] in {where_clause}")
                        elif i < len(tokens):
                            # 检查值是否是数字或字符串字面量
                            if re.match(r'^-?\d+(\.\d+)?$', tokens[i]) or tokens[i].startswith('STRING_LITERAL_'):
                                i += 1
                            else:
                                raise ValueError(f"Invalid query syntax: invalid value {tokens[i]} in {where_clause}")
                        else:
                            raise ValueError(f"Invalid query syntax: missing value after operator in {where_clause}")
                    else:
                        raise ValueError(f"Invalid query syntax: invalid operator after field name in {where_clause}")
                    continue
                
                # 检查是否是数字字面量
                if re.match(r'^-?\d+(\.\d+)?$', token):
                    i += 1
                    continue
                
                raise ValueError(f"Invalid query syntax: unexpected token {token} in {where_clause}")
            
            # 7. 验证函数调用
            for func_call in self._function_calls:
                if 'json_extract' in func_call.lower():
                    # 修改正则表达式以适应我们的字符串替换
                    match = re.match(r'json_extract\s*\(([^,]+),\s*(?:STRING_LITERAL_\d+|[\'"][^\'\"]+[\'\"])\)', func_call, re.IGNORECASE)
                    if not match:
                        raise ValueError(f"Invalid query syntax: invalid json_extract syntax in {func_call}")
                elif 'cast' in func_call.lower():
                    match = re.match(r'cast\s*\(([^,]+)\s+as\s+(integer|real|text)\)', func_call, re.IGNORECASE)
                    if not match:
                        raise ValueError(f"Invalid query syntax: invalid CAST syntax in {func_call}")
            
            return True
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Invalid query syntax: {str(e)}")

    def query(self, query_filter: str = None, return_ids_only: bool = True, limit: int = None, offset: int = None) -> Union[List[int], List[Dict[str, Any]]]:
        """
        Query records using SQL-like filter syntax.

        Parameters:
            query_filter: str
                SQL-like filter condition. Examples:
                - "age > 18"
                - "name LIKE '%John%'"
                - "age > 18 AND city = 'New York'"
                - "json_extract(data, '$.address.city') = 'Beijing'"
                - "1=1"  # 返回所有记录
            return_ids_only: bool
                If True, only return IDs. If False, return complete records.
            limit: int
                Maximum number of records to return.
            offset: int
                Number of records to skip.

        Returns:
            Union[List[int], List[Dict[str, Any]]]: List of IDs or complete records.

        Raises:
            ValueError: 当查询语法无效时抛出
        """
        try:
            # 验证查询语法
            if query_filter and not query_filter.strip():
                raise ValueError("Invalid query syntax: empty query")

            # 构建基本查询
            if return_ids_only:
                base_query = "SELECT _id FROM records"
            else:
                # 获取所有字段名（除了_id）
                fields = self.list_fields()
                if not fields:
                    return []
                fields_str = ', '.join(f'"{field}"' for field in fields)
                base_query = f"SELECT {fields_str} FROM records"

            # 处理查询条件
            parameters = []
            if query_filter:
                # 解析查询
                parsed = self._parse_query(query_filter)

                # 验证 WHERE 子句
                if parsed['where'] != '1=1' and not self._validate_where(parsed['where']):
                    raise ValueError(f"Invalid query syntax: {query_filter}")

                # 使用AST解析器解析WHERE子句
                if parsed['where'] != '1=1':
                    try:
                        self.generator.reset()  # 重置生成器状态
                        ast = self.parser.parse(parsed['where'])
                        where_clause = self.generator.generate(ast)
                        parameters = self.generator.get_parameters()
                        print(f"Debug - SQL Query: {base_query} WHERE {where_clause}")
                        print(f"Debug - Parameters: {parameters}")
                    except Exception as e:
                        raise ValueError(f"Invalid query syntax: {str(e)}")

                    # 添加WHERE子句
                    base_query += f" WHERE {where_clause}"

                # 添加ORDER BY子句
                if parsed['order']:
                    if not self._validate_order_by(parsed['order']):
                        raise ValueError("Invalid ORDER BY clause")
                    base_query += f" {parsed['order']}"

            # 添加分页
            if limit is not None:
                if not isinstance(limit, int) or limit < 0:
                    raise ValueError("Invalid LIMIT value")
                base_query += f" LIMIT {limit}"
                if offset is not None:
                    if not isinstance(offset, int) or offset < 0:
                        raise ValueError("Invalid OFFSET value")
                    base_query += f" OFFSET {offset}"

            # 执行查询
            cursor = self.storage.conn.cursor()
            try:
                if parameters:
                    results = cursor.execute(base_query, parameters).fetchall()
                else:
                    results = cursor.execute(base_query).fetchall()

                if return_ids_only:
                    return [row[0] for row in results]
                else:
                    # 构建结果字典
                    records = []
                    fields = self.list_fields()
                    for row in results:
                        record = {}
                        for i, value in enumerate(row):
                            if value is not None:
                                if isinstance(value, str):
                                    try:
                                        parsed_value = json.loads(value)
                                        record[fields[i]] = parsed_value
                                    except json.JSONDecodeError:
                                        record[fields[i]] = value
                                else:
                                    record[fields[i]] = value
                        records.append(record)
                    return records
            except Exception as e:
                if "no such column" in str(e):
                    return [] if return_ids_only else []
                raise ValueError(f"Invalid query syntax: {query_filter}")
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Query failed: {str(e)}")

    def list_fields(self) -> List[str]:
        """
        获取所有可用字段列表。

        Returns:
            List[str]: 字段名列表
        """
        try:
            cursor = self.storage.conn.cursor()
            results = cursor.execute("SELECT field_name FROM fields_meta").fetchall()
            return [row[0] for row in results]
        except Exception as e:
            raise ValueError(f"Failed to list fields: {str(e)}")

    def get_field_type(self, field_name: str) -> str:
        """
        获取字段类型。

        Parameters:
            field_name: str
                字段名称

        Returns:
            str: 字段类型
        """
        try:
            cursor = self.storage.conn.cursor()
            result = cursor.execute(
                "SELECT field_type FROM fields_meta WHERE field_name = ?",
                [field_name]
            ).fetchone()
            return result[0] if result else None
        except Exception as e:
            raise ValueError(f"Failed to get field type: {str(e)}")

    def create_field_index(self, field_name: str):
        """
        为指定字段创建索引。

        Parameters:
            field_name: str
                字段名称
        """
        try:
            cursor = self.storage.conn.cursor()
            # 检查字段是否存在
            field_exists = cursor.execute(
                "SELECT 1 FROM fields_meta WHERE field_name = ?",
                [field_name]
            ).fetchone()
            
            if field_exists:
                # 创建索引，使用引号包裹字段名
                index_name = f"idx_{field_name}"
                quoted_field_name = self._quote_identifier(field_name)
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON records({quoted_field_name})
                """)
                cursor.execute("ANALYZE")
            else:
                raise ValueError(f"Field {field_name} does not exist")
        except Exception as e:
            raise ValueError(f"Failed to create index: {str(e)}")

    def retrieve(self, id_: int) -> Dict[str, Any]:
        """
        Retrieve a single record by ID.

        Parameters:
            id_: int
                The ID of the record to retrieve.

        Returns:
            Dict[str, Any]: The record if found, None otherwise.
        """
        try:
            # 获取所有字段名（除了_id）
            fields = self.list_fields()
            if not fields:
                return None
            fields_str = ', '.join(f'"{field}"' for field in fields)
            
            result = self.storage.conn.execute(
                f"SELECT {fields_str} FROM records WHERE _id = ?",
                [id_]
            ).fetchone()
            
            if result:
                # 构建结果字典，不包含内部ID
                record = {}
                for i, value in enumerate(result):
                    if value is not None:
                        # 尝试解析JSON字符串
                        if isinstance(value, str):
                            try:
                                parsed_value = json.loads(value)
                                record[fields[i]] = parsed_value
                            except json.JSONDecodeError:
                                record[fields[i]] = value
                        else:
                            record[fields[i]] = value
                return record
            return None
        except Exception as e:
            raise ValueError(f"Failed to retrieve record: {str(e)}")

    def retrieve_many(self, ids: List[int]) -> List[Dict[str, Any]]:
        """
        Retrieve multiple records by their IDs.

        Parameters:
            ids: List[int]
                List of record IDs to retrieve.

        Returns:
            List[Dict[str, Any]]: List of retrieved records.
        """
        if not ids:
            return []

        try:
            cursor = self.storage.conn.cursor()
            
            # 获取所有字段名（除了_id）
            fields = self.list_fields()
            if not fields:
                return []
            fields_str = ', '.join(f'"{field}"' for field in fields)

            # 使用参数化查询
            placeholders = ','.join('?' * len(ids))
            query = f"""
                SELECT _id, {fields_str}
                FROM records
                WHERE _id IN ({placeholders})
                ORDER BY _id
            """
            
            results = cursor.execute(query, ids).fetchall()

            # 构建ID到记录的映射
            id_to_record = {}
            for row in results:
                record_id = row[0]  # 第一列是_id
                if any(row[1:]):  # 检查是否有任何非空字段
                    record = {}
                    for i, value in enumerate(row[1:], 0):  # 跳过_id列
                        if value is not None:
                            # 尝试解析JSON字符串
                            if isinstance(value, str):
                                try:
                                    parsed_value = json.loads(value)
                                    record[fields[i]] = parsed_value
                                except json.JSONDecodeError:
                                    record[fields[i]] = value
                            else:
                                record[fields[i]] = value
                    id_to_record[record_id] = record

            # 按照输入ID的顺序返回记录
            return [id_to_record.get(id_) for id_ in ids]

        except Exception as e:
            raise ValueError(f"Failed to retrieve records: {str(e)}")

    def _create_temp_indexes(self, field: str, json_path: str):
        """
        为 JSON 路径创建临时索引。

        Parameters:
            field: str
                字段名
            json_path: str
                JSON 路径
        """
        try:
            cursor = self.storage.conn.cursor()
            
            # 生成安全的索引名
            safe_path = json_path.replace('$', '').replace('.', '_').replace('[', '_').replace(']', '_')
            index_name = f"idx_json_{field}_{safe_path.strip('_')}"
            
            # 创建索引
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS {index_name}
                ON records(json_extract({field}, ?))
            """, (json_path,))
            
            # 分析新创建的索引
            cursor.execute("ANALYZE")
        except Exception as e:
            # 如果创建索引失败，记录错误但继续执行
            print(f"Warning: Failed to create JSON index: {str(e)}")

    def _validate_json_path(self, json_path: str) -> bool:
        """
        验证 JSON 路径语法。

        Parameters:
            json_path: str
                JSON 路径表达式，例如 '$.name' 或 '$.address.city'

        Returns:
            bool: 路径语法是否有效
        """
        if not json_path:
            return False

        # 基本语法检查
        if not json_path.startswith('$'):
            return False

        # 检查路径组件
        parts = json_path[2:].split('.')
        for part in parts:
            if not part:
                return False
            # 检查数组访问语法
            if '[' in part:
                if not part.endswith(']'):
                    return False
                array_part = part.split('[')[1][:-1]
                if not array_part.isdigit():
                    return False
            # 检查普通字段名
            else:
                if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', part):
                    return False

        return True
