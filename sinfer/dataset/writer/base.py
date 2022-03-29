import os.path as osp
from typing import Dict


class BaseWriter(object):
    def __init__(self, config: Dict, ext: str="tif") -> None:
        file_split = osp.split(config["file_path"])
        self.save_path = osp.join(file_split[0], osp.splitext(file_split[-1])[0] + "_output." + ext)
        self.width = config["width"]
        self.height = config["height"]
        self.proj = config["proj"]
        self.geotf = config["geotf"]
        self.block_size = config["block_size"]
        self.overlap = config["overlap"]
        self.num_block = config["num_block"]

    def write(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError