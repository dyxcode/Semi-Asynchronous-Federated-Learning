from pathlib import Path
import torch
from transformers import AutoImageProcessor
import numpy as np

import sys
from pathlib import Path
current_dir = Path().absolute()
if current_dir.name != 'knowledge_distillation':
    sys.path.append(str(current_dir / 'knowledge_distillation'))

from manager import TxtManager

class DinoV2Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        
    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = inputs.pixel_values.squeeze(0) # (1,3,224,224) => (3,224,224)
        return inputs, label

    def __len__(self):
        return len(self.dataset)
    
class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, logits_path: str, topk, write):
        super().__init__()
        self.dataset = dataset
        self.logits_path = Path(logits_path).absolute()
        self.topk = topk
        self.write_mode = write
        self.keys = self._get_keys()
        self._manager = None

    def __getitem__(self, index: int):
        if self.write_mode:
            return self.__getitem_for_write(index)
        return self.__getitem_for_read(index)

    def __getitem_for_write(self, index: int):
        key = self.keys[index]
        item = self.dataset[index]
        return (item, key)

    def __getitem_for_read(self, index: int):
        key = self.keys[index]
        logits_index, logits_value = self._get_saved_logits(key)
        item = self.dataset[index]
        return (item, (logits_index, logits_value))

    def _get_saved_logits(self, key: str):
        manager = self.get_manager()
        bstr: bytes = manager.read(key)
        # parse the logits index and value
        # copy logits_index and logits_value to avoid warning of written flag from PyTorch
        logits_index = np.frombuffer(
            bstr[:self.topk * 4], dtype=np.int32).copy()
        bstr = bstr[self.topk * 4:]
        logits_value = np.frombuffer(
            bstr[:self.topk * 4], dtype=np.float32).copy()
        return logits_index, logits_value

    def _build_manager(self, logits_path: str):
        # topk * [idx, value] * 4 bytes for logits
        item_size = self.topk * 2 * 4
        return TxtManager(logits_path, item_size)

    def get_manager(self):
        if (self._manager == None):
            self._manager = self._build_manager(self.logits_path)
        return self._manager

    def __len__(self):
        return len(self.dataset)

    def _get_keys(self):
        return [str(i) for i in range(len(self))]