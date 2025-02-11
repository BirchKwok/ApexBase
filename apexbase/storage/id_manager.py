
class IDManager:
    """ID manager"""
    
    def __init__(self, storage):
        """Initialize the ID manager"""
        self.storage = storage
        self._id = {}

    def get_next_id(self, table_name: str):
        """Get the next ID"""
        if table_name not in self._id:
            self._id[table_name] = self.storage._get_next_id(table_name)
        return self._id[table_name]
    
    def auto_increment(self, table_name: str):
        """Auto increment the ID"""
        if table_name not in self._id:
            self._id[table_name] = self.storage._get_next_id(table_name)
        self._id[table_name] += 1
    
    def reset_last_id(self, table_name: str):
        """Reset the last ID"""
        self._id[table_name] = self.storage._get_next_id(table_name)

    def current_id(self, table_name: str):
        """Get the current ID"""
        if table_name not in self._id:
            self._id[table_name] = self.storage._get_next_id(table_name) - 1
        return self._id[table_name] - 1
        