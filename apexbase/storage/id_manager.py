
class IDManager:
    """ID管理器"""
    
    def __init__(self, storage):
        """初始化ID管理器"""
        self.storage = storage
        self._id = {}

    def get_next_id(self, table_name: str):
        """获取下一个ID"""
        if table_name not in self._id:
            self._id[table_name] = self.storage._get_next_id(table_name)
        return self._id[table_name]
    
    def auto_increment(self, table_name: str):
        """自动增加ID"""
        if table_name not in self._id:
            self._id[table_name] = self.storage._get_next_id(table_name)
        self._id[table_name] += 1
    
    def reset_last_id(self, table_name: str):
        """重置最后一个ID"""
        self._id[table_name] = self.storage._get_next_id(table_name)

    def current_id(self, table_name: str):
        """获取当前ID"""
        if table_name not in self._id:
            self._id[table_name] = self.storage._get_next_id(table_name) - 1
        return self._id[table_name] - 1
        