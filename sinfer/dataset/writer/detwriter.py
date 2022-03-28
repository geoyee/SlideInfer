import numpy as np
from operator import itemgetter
from typing import List, Dict, Union
from .base import BaseWriter

try:
    from osgeo import gdal
except:
    import gdal


class DetWriter(BaseWriter):
    def __init__(self, config: Dict) -> None:
        super(DetWriter, self).__init__(config, "geojson")

    def write(self) -> None:
        self.save_list = sorted(self.save_list, key=itemgetter("start"))