from typing import List, Union, Dict, Any
import json
import re


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
        # SQL语法组件
        self.sql_components = {
            'operators': {'=', '>', '<', '>=', '<=', '!=', 'LIKE', 'IN', 'IS', 'NOT', 'NULL'},
            'logical': {'AND', 'OR'},
            'order': {'ORDER', 'BY', 'ASC', 'DESC'},
            'special': {'1=1'},  # 特殊条件
            'functions': {'CAST', 'AS', 'INTEGER', 'REAL', 'TEXT', 'JSON_EXTRACT'}  # 函数和类型
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

    def _parse_query(self, query: str) -> dict:
        """
        解析查询语句，将其分解为组件。

        Parameters:
            query: str
                查询语句

        Returns:
            dict: 包含解析后的查询组件
        """
        if not query:
            return {'where': None, 'order': None}

        # 处理特殊查询
        if query.upper() == '1=1':
            return {'where': '1=1', 'order': None}

        # 分离ORDER BY子句
        order_match = re.search(r'\bORDER\s+BY\s+(.+)$', query, re.IGNORECASE)
        where_clause = query
        order_clause = None

        if order_match:
            where_clause = query[:order_match.start()].strip()
            order_clause = order_match.group()

        return {
            'where': where_clause if where_clause and where_clause != '1=1' else '1=1',
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
        """
        if not where_clause or where_clause == '1=1':
            return True

        # 预处理：将表达式转换为标准形式
        # 1. 保护字符串字面量
        def preserve_string(match):
            return f"STRING_LITERAL_{len(self._string_literals)}"
        
        self._string_literals = []
        cleaned = where_clause
        cleaned = re.sub(r"'([^']*)'", lambda m: (self._string_literals.append(m.group(1)), preserve_string(m))[1], cleaned)
        
        # 2. 保护函数调用
        def preserve_function(match):
            func_name = match.group(1).upper()
            args = match.group(2)
            return f"{func_name}_CALL_{len(self._function_calls)}"
        
        self._function_calls = []
        cleaned = re.sub(r'(json_extract|cast)\s*\(([^)]+)\)', lambda m: (self._function_calls.append(m.group(0)), preserve_function(m))[1], cleaned, flags=re.IGNORECASE)
        
        # 3. 将其余部分转为大写
        cleaned = cleaned.upper()
        
        # 4. 标准化空白字符
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # 5. 分割成 tokens
        tokens = cleaned.split()
        
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
                continue
            
            # 检查是否是操作符
            if token in self.sql_components['operators']:
                i += 1
                continue
            
            # 检查是否是逻辑操作符
            if token in self.sql_components['logical']:
                i += 1
                continue
            
            # 检查是否是数字
            if token.isdigit() or token == 'NULL':
                i += 1
                continue
            
            # 检查是否是标识符
            if re.match(r'^[A-Z_][A-Z0-9_]*$', token):
                i += 1
                continue
            
            # 如果都不匹配，说明是无效的 token
            return False
        
        # 7. 验证函数调用
        for func_call in self._function_calls:
            if 'json_extract' in func_call.lower():
                match = re.match(r'json_extract\s*\(([^,]+),\s*\'([^\']+)\'\)', func_call, re.IGNORECASE)
                if not match:
                    return False
                field = match.group(1).strip()
                path = match.group(2)
                if not self._validate_json_path(path):
                    return False
            elif 'cast' in func_call.lower():
                match = re.match(r'cast\s*\(([^)]+)\)\s+as\s+(integer|real|text)', func_call, re.IGNORECASE)
                if not match:
                    return False
                expr = match.group(1).strip()
                type_ = match.group(2).upper()
                if not expr or type_ not in {'INTEGER', 'REAL', 'TEXT'}:
                    return False
        
        return True

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
                - "1=1 ORDER BY age DESC"  # 按年龄降序排序
                If None, returns all records.
            return_ids_only: bool
                If True, only return IDs. If False, return complete records.
            limit: int
                Maximum number of records to return.
            offset: int
                Number of records to skip.

        Returns:
            Union[List[int], List[Dict[str, Any]]]: List of IDs or complete records.
        """
        try:
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
            if query_filter:
                # 解析查询
                parsed = self._parse_query(query_filter)
                
                # 验证WHERE子句
                if not self._validate_where(parsed['where']):
                    raise ValueError(f"Invalid query syntax: {query_filter}")
                
                # 验证ORDER BY子句
                if parsed['order'] and not self._validate_order_by(parsed['order']):
                    raise ValueError(f"Invalid query syntax: {query_filter}")

                # 提取并验证 JSON 路径
                json_paths = re.findall(r'json_extract\s*\(([^,]+),\s*\'([^\']+)\'\)', parsed['where'])
                for field, path in json_paths:
                    field = field.strip()
                    if not self._validate_json_path(path):
                        raise ValueError(f"Invalid JSON path: {path}")
                    # 创建临时索引以优化查询
                    self._create_temp_indexes(field, path)

                # 验证字段是否存在
                if parsed['where'] != '1=1':
                    # 提取查询中的字段名（排除json_extract函数）
                    where_no_json = re.sub(r'json_extract\s*\([^)]+\)', '', parsed['where'])
                    field_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*[=<>!]'
                    fields_in_query = re.findall(field_pattern, where_no_json)
                    
                    # 获取所有可用字段
                    available_fields = set(self.list_fields())
                    
                    # 检查每个字段是否存在
                    for field in fields_in_query:
                        if field not in available_fields:
                            return [] if return_ids_only else []

                # 添加WHERE子句
                if parsed['where']:
                    base_query += f" WHERE {parsed['where']}"

                # 添加ORDER BY子句
                if parsed['order']:
                    base_query += f" {parsed['order']}"

            # 添加分页
            if limit is not None:
                base_query += f" LIMIT {limit}"
                if offset is not None:
                    base_query += f" OFFSET {offset}"

            # 执行查询
            cursor = self.storage.conn.cursor()
            try:
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
