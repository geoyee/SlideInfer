import os.path as osp
import paddle
from paddle.nn import Layer
from tqdm import tqdm
from typing import List, Union
from .baseinfer import BaseSlider
from ..dataset import RasterLoader, SegWriter


class SegSlider(BaseSlider):
    def __init__(self, model: Layer) -> None:
        """ Slide infer about segmentation.

        Args:
            model (Layer): Model of PaddleSeg.
        """
        super(SegSlider, self).__init__(model)
        
    def __call__(self, path: str) -> None:
        dataloader = RasterLoader(path, self.block_size, self.overlap, self.mean, self.std)
        datawriter = SegWriter(dataloader.config)
        for data in tqdm(dataloader):
            img = data["block"]
            start = data["start"]
            img = paddle.to_tensor(img.transpose(2, 0, 1)[None])
            pred = self.model(img)[0]
            block = paddle.argmax(pred, axis=1).squeeze().numpy().astype("uint8")
            datawriter.write(block, start)
        datawriter.close()
        print("[Finshed] The file saved {0}.".format(osp.normpath(datawriter.save_path)))

    def ready(self, 
              block_size: Union[List[int], int]=512,
              overlap: Union[List[int], int]=32,
              mean: Union[List[float], None]=[0.5, 0.5, 0.5], 
              std: Union[List[float], None]=[0.5, 0.5, 0.5]) -> None:
        """ Ready.

        Args:
            block_size (Union[List[int], int], optional): Size of image's block. Defaults to 512.
            overlap (Union[List[int], int], optional): Overlap between two images. Defaults to 32.
            mean (Union[List[float], None], optional): Mean of normalize. Defaults to [0.5, 0.5, 0.5].
            std (Union[List[float], None], optional): Std of normalize. Defaults to [0.5, 0.5, 0.5].
        """
        super(SegSlider, self).ready(block_size, mean, std)
        self.overlap = overlap