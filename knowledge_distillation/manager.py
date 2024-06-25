from pathlib import Path

class Writer():
    def __init__(self, path: Path):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)

        self.keys_file = (self.path / 'keys.txt').open('w')
        self.values_file = (self.path / 'values.bin').open('wb')

    def write(self, key: str, value: bytes):
        self.keys_file.write(key + '\n')
        self.values_file.write(value)

    def __del__(self):
        self.keys_file.close()
        self.values_file.close()

class Reader:
    def __init__(self, path: Path, item_size: int):
        self.item_size = item_size
        self.path = path
        self.keys = self.load_keys()
        self.values_file = self.open_values_file()

    def load_keys(self):
        keys = {}
        keys_fname = self.path / 'keys.txt'
        with open(keys_fname, 'r') as keys_file:
            for i, key in enumerate(keys_file.readlines()):
                key = key.strip()
                keys[key] = i
        return keys

    def open_values_file(self):
        values_fname = self.path / 'values.bin'
        return open(values_fname, 'rb')

    def read(self, key: str) -> bytes:
        idx = self.keys.get(key)
        if idx is None:
            raise KeyError(key)
        self.values_file.seek(self.item_size * idx)
        return self.values_file.read(self.item_size)
    
    def __del__(self):
        self.values_file.close()


class TxtManager:
    def __init__(self, path: Path, item_size: int):
        self.path = path
        self.writer = None
        self.reader = None
        self.item_size = item_size

    def write(self, key: str, value: bytes) -> bool:
        if self.writer is None:
            self.writer = Writer(self.path)
        return self.writer.write(key, value)

    def read(self, key: str) -> bytes:
        if self.reader is None:
            self.reader = Reader(self.path, self.item_size)
        return self.reader.read(key)