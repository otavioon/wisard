class Filter:
    def add_member(self, value, inc_val: int = 1):
        raise NotImplementedError

    def check_membership(self, value, soft_error_rate: float = 0.0):
        raise NotImplementedError

    def set_bleaching(self, bleach: int = 1):
        raise NotADirectoryError

    def __contains__(self, key):
        return self.check_membership(key)