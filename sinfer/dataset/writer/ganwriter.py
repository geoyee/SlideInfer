import numpy as np
from typing import List, Dict
from .base import BaseWriter

try:
    from osgeo import gdal
except:
    import gdal


class GanWriter(BaseWriter):
    def __init__(self, config: Dict, scale: int) -> None:
        super(GanWriter, self).__init__(config, "tif")
        driver = gdal.GetDriverByName("GTiff")
        self.scale = scale
        self.dst_ds = driver.Create(
            self.save_path, int(scale * self.width), 
            int(scale * self.height), 3, gdal.GDT_UInt16)
        self.geotf = list(self.geotf)
        self.geotf[1] = self.geotf[1] / scale
        self.geotf[5] = self.geotf[5] / scale
        self.geotf = tuple(self.geotf)
        self.dst_ds.SetGeoTransform(self.geotf)
        self.dst_ds.SetProjection(self.proj)
        self.bands = [self.dst_ds.GetRasterBand(i + 1) for i in range(3)]

    def write(self, block: np.ndarray, start: List[int]) -> None:
        bw, bh = block.shape[:2]
        width = self.dst_ds.RasterXSize
        height = self.dst_ds.RasterYSize
        xoff = start[0] * self.scale
        yoff = start[1] * self.scale
        xsize = xoff + bw
        ysize = yoff + bh
        xsize = int(width - xoff) if xsize > width else int(bw)
        ysize = int(height - yoff) if ysize > height else int(bh)
        for i, band in enumerate(self.bands):
            band.WriteArray(block[:ysize, :xsize, i], int(xoff), int(yoff))
        self.dst_ds.FlushCache()

    def close(self) -> None:
        self.dst_ds = None