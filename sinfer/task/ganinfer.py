import os.path as osp
import numpy as np
import paddle
from paddle.nn import Layer
from tqdm import tqdm
from typing import List, Union
from .baseinfer import BaseSlider
from ..dataset import RasterLoader, GanWriter


class GanSlider(BaseSlider):
    def __init__(self, model: Layer) -> None:
        """ Slide infer about gan.

        Args:
            model (Layer): Model of PaddleGAN.
        """
        super(GanSlider, self).__init__(model)
        
    def __call__(self, path: str) -> None:
        dataloader = RasterLoader(path, self.block_size, 0, False, self.mean, self.std)
        datawriter = GanWriter(dataloader.config, self.scale)
        for data in tqdm(dataloader):
            img = data["block"]
            start = data["start"]
            img = paddle.to_tensor(img.transpose(2, 0, 1)[None])
            with paddle.no_grad():
                pred = self.model(img)[-1]
            block = self.__deNormal(pred.squeeze().numpy())
            datawriter.write(block, start)
        datawriter.close()
        print("[Finshed] The file saved {0}.".format(osp.normpath(datawriter.save_path)))

    def ready(self, 
              scale: int=4,
              block_size: Union[List[int], int]=64,
              mean: Union[List[float], None]=[0, 0, 0], 
              std: Union[List[float], None]=[1, 1, 1]) -> None:
        """ Ready.

        Args:
            scale (int, optional): Magnification. Defaults to 4.
            block_size (Union[List[int], int], optional): Size of image's block. Defaults to 64.
            mean (Union[List[float], None], optional): Mean of normalize. Defaults to [0, 0, 0].
            std (Union[List[float], None], optional): Std of normalize. Defaults to [1, 1, 1].
        """
        super(GanSlider, self).ready(block_size, mean, std)
        self.scale = scale

    def __deNormal(self, img: np.ndarray) -> np.ndarray:
        img = img.transpose((1, 2, 0))
        img = img * self.std + self.mean
        img = img.clip(0, 255).astype("uint8")
        return img
