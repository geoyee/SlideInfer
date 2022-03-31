from paddle.nn import Layer
from typing import List, Union


class BaseSlider(object):
    def __init__(self, model: Layer) -> None:
        self.model = model
        self.model.eval()
        self.ready()
        
    def __call__(self) -> None:
        raise NotImplementedError()

    def ready(self, 
              block_size: Union[List[int], int]=512,
              mean: Union[List[float], None]=[0.5, 0.5, 0.5], 
              std: Union[List[float], None]=[0.5, 0.5, 0.5]) -> None:
        self.block_size = block_size
        self.mean = mean
        self.std = std