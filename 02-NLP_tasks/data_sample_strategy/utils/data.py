import os
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from datasets import load_dataset
from torch.utils.data import Dataset
from typing import Any, List, Tuple, Dict
from transformers import default_data_collator,DefaultDataCollator

class TrainDatasetForOrder(Dataset):
    def __init__(self, dataset_dir: str, cache_dir: str = 'cache_data') -> None:
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.cache_dir = cache_dir

        self.spec_data = self.load_data(type_name="specdata")
        self.norm_data = self.load_data(type_name="normdata")

        self.len_spec = len(self.spec_data)
        self.len_norm = len(self.norm_data)
        self.len_total = self.len_spec + self.len_norm

    def get_all_datapath(self, dir_name: str) -> List[str]:
        all_file_path = []

        for root, dir, file_name in os.walk(dir_name):
            for item in file_name:
                if item.endswith('json'):
                    standard_path = os.path.join(root, item)
                    all_file_path.append(standard_path)
        return all_file_path
    
    def get_all_datapath2(self, dir_name: str) -> List[str]:
        if isinstance(dir_name, Path):
            dir_name = str(dir_name)
        all_file_list = []

        for root, dir, file_name in os.walk(dir_name):
            for temp_file in file_name:
                standard_path = f"{root}/{temp_file}"

                all_file_list.append(standard_path)

        return all_file_list
    
    def load_data_from_path(self, data_path: Optional[str] = None, cache_dir: Optional[str] = "cache_data") -> Dataset:
        all_file_list =  self.get_all_datapath2(data_path)
        data_file ={'train': all_file_list}
        exension = all_file_list[0].split('.')[-1]

        raw_dataset = load_dataset(exension, data_files=data_file, cache_dir=cache_dir)['train']
        return raw_dataset
    
    def load_data(self, type_name: str) -> Dataset:
        data = self.load_data_from_path(
            str(self.dataset_dir.joinpath(type_name)), self.cache_dir
        )
        return data


    def __len__(self):
        return self.len_total

    def __getitem__(self, index):
        if index < self.len_spec:
            tempdata = self.spec_data[index]
            tempdata = torch.tensor([100, int(tempdata["text"])])
            return tempdata
        else:
            tempdata = self.norm_data[index - self.len_spec]
            tempdata = torch.tensor([200, int(tempdata["text"])])
            return tempdata

@dataclass
class GroupCollator():
    # def __len__(self):
    #     return 2000
    
    def __call__(self, features: List):
        temp_data = torch.concatenate([i.reshape(1,-1) for i in features], dim=0)
        temp_data = {"batch" : temp_data}
        return temp_data
    

if __name__ == "__main__":
    dataset_test = TrainDatasetForOrder(
        dataset_dir="D:/AI/NLP/LLM-RoadMap/02-NLP_tasks/data/test_data"
    )

    groupcollator = GroupCollator()
    # groupcollator = default_data_collator()

    res = groupcollator([dataset_test[i] for i in [1, 130, 999]], )

    print(res)