import numpy as np
from math import ceil
from typing import List, Dict
from .base import BaseWriter

try:
    from osgeo import gdal
except:
    import gdal


class ClsWriter(BaseWriter):
    def __init__(self, config: Dict) -> None:
        super(ClsWriter, self).__init__(config, "tif")
        driver = gdal.GetDriverByName("GTiff")
        self.dst_ds = driver.Create(
            self.save_path, ceil(self.width / self.block_size[0]), 
            ceil(self.height / self.block_size[1]), 1, gdal.GDT_UInt16)
        self.geotf = list(self.geotf)
        self.geotf[1] = self.geotf[1] * self.block_size[0]
        self.geotf[5] = self.geotf[5] * self.block_size[1]
        self.geotf = tuple(self.geotf)
        self.dst_ds.SetGeoTransform(self.geotf)
        self.dst_ds.SetProjection(self.proj)
        self.band = self.dst_ds.GetRasterBand(1)

    def write(self, block: np.ndarray, start: List[int]) -> None:
        xoff = start[0] / self.block_size[0]
        yoff = start[1] / self.block_size[1]
        self.band.WriteArray(block, int(xoff), int(yoff))
        self.dst_ds.FlushCache()

    def close(self) -> None:
        self.dst_ds = None