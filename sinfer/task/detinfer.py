import os.path as osp
import paddle
from paddle.nn import Layer
from tqdm import tqdm
from typing import List, Union
from ..dataset import RasterLoader, DetWriter


class DetSlider(object):
    def __init__(self, model: Layer) -> None:
        """ Slide infer about detection.

        Args:
            model (Layer): Model of PaddleDetection.
        """
        self.model = model
        self.model.eval()
        self.ready()
        
    def __call__(self, path) -> None:
        dataloader = RasterLoader(path, self.block_size, self.overlap, self.mean, self.std)
        datawriter = DetWriter(dataloader.config)
        datawriter.draw_threshold = self.draw_threshold
        for data in tqdm(dataloader):
            img = data["block"]
            start = data["start"]
            img = paddle.to_tensor(img.transpose(2, 0, 1)[None])
            inputs = {
                "image": img,
                "im_shape": paddle.to_tensor(img.shape[2:], dtype="float32").reshape([1, 2]),
                "scale_factor": paddle.to_tensor([1., 1.], dtype="float32")
                }
            pred = self.model(inputs)
            if len(pred["bbox_num"].numpy().tolist()) != 0:
                datawriter.write(pred["bbox"].numpy(), start)
        datawriter.close()
        print("[Finshed] The file saved {0}.".format(osp.normpath(datawriter.save_path)))

    def ready(self, 
              draw_threshold: float=0.5,
              block_size: Union[List[int], int]=608,
              overlap: Union[List[int], int]=32,
              mean: Union[List[float], None]=[0.485, 0.456, 0.406], 
              std: Union[List[float], None]=[0.229, 0.224, 0.225]) -> None:
        """ Ready.

        Args:
            draw_threshold (float, optional): Threshold of bbox to display. Defaults to 0.5.
            block_size (Union[List[int], int], optional): Size of image's block. Defaults to 608.
            overlap (Union[List[int], int], optional): Overlap between two images. Defaults to 32.
            mean (Union[List[float], None], optional): Mean of normalize. Defaults to [0.485, 0.456, 0.406].
            std (Union[List[float], None], optional): Std of normalize. Defaults to [0.229, 0.224, 0.225].
        """
        self.draw_threshold = draw_threshold
        self.block_size = block_size
        self.overlap = overlap
        self.mean = mean
        self.std = std