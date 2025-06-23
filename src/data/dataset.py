import torch
import numpy as np

from torch import Tensor
from torch.utils import data
from typing import Union, Tuple


class Dataset(data.Dataset):
    def __init__(self,
                 Xs: Union[Tensor, np.ndarray],
                 ys: Union[Tensor, np.ndarray],
                 dtype: torch.dtype = torch.float32) -> None:
        super(Dataset, self).__init__()
        assert len(Xs) == len(ys)
        self.Xs = Xs if Xs.dtype==dtype else torch.tensor(Xs).to(dtype=dtype)
        self.ys = ys if ys.dtype==dtype else torch.tensor(ys).to(dtype=dtype)
        self.length = len(Xs)
    
    def __getitem__(self,
                    index: int) -> Tuple[Tensor, Tensor]:
        X = self.Xs[index, :]
        y = self.ys[index]

        return X, y
    
    def __len__(self):
        return self.length