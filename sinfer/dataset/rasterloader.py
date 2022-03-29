import numpy as np
from typing import List, Dict, Union

try:
    from osgeo import gdal
except:
    import gdal


class RasterLoader(object):
    def __init__(self,
                 path: str,
                 block_size: Union[List[int], int]=512,
                 overlap: Union[List[int], int]=32,
                 mean: Union[List[float], None]=[0.5, 0.5, 0.5], 
                 std: Union[List[float], None]=[0.5, 0.5, 0.5]) -> None:
        """ Dataloadr about geo-raster.

        Args:
            path (str): Path of big-geo-image.
            block_size (Union[List[int], int], optional): Size of image's block. Defaults to 512.
            overlap (Union[List[int], int], optional): Overlap between two images. Defaults to 32.
            mean (Union[List[float], None], optional): Mean of normalize. Defaults to [0.5, 0.5, 0.5].
            std (Union[List[float], None], optional): Std of normalize. Defaults to [0.5, 0.5, 0.5].

        Raises:
            ValueError: Can't read iamge from this path. 
        """
        self._src_data = gdal.Open(path)
        if self._src_data is None:
            raise ValueError("Can't read iamge from file {0}.".format(path))
        self.path = path
        if isinstance(block_size, int):
            self.block_size = [block_size, block_size]
        else:
            self.block_size = list(block_size)
        if isinstance(overlap, int):
            self.overlap = [overlap, overlap]
        else:
            self.overlap = list(overlap)
        if len(self.block_size) != 2 or len(self.overlap) != 2:
            raise IndexError(
                "Lenght of `block_size`/`overlap` must be 2, not {0}/{1}.".format(
                    len(self.block_size), len(self.overlap)))
        self.mean = list(mean) if mean is not None else None
        self.std = list(std) if std is not None else None
        if len(self.mean) != 3 or len(self.std) != 3:
            raise IndexError(
                "Lenght of `mean`/`std` must be 3, not {0}/{1}.".format(
                    len(self.mean), len(self.mean)))
        self.__getInfos()
        self.__getStart()

    def __getitem__(self, index) -> np.ndarray:
        start_loc = self._start_list[index]
        return self.__getBlock(start_loc)

    def __len__(self) -> int:
        return len(self._start_list)

    @property
    def config(self) -> Dict:
        return {
            "file_path": self.path,
            "width": self.width,
            "height": self.height,
            "proj": self.proj,
            "geotf": self.geotf,
            "block_size": self.block_size,
            "overlap": self.overlap,
            "num_block": self.__len__,
        }

    def __getInfos(self) -> None:
        self.bands = self._src_data.RasterCount
        self.width = self._src_data.RasterXSize
        self.height = self._src_data.RasterYSize
        self.geotf = self._src_data.GetGeoTransform()
        self.proj = self._src_data.GetProjection()

    def __getStart(self) -> None:
        self._start_list = []
        step_r = self.block_size[1] - self.overlap[1]
        step_c = self.block_size[0] - self.overlap[0]
        for r in range(0, self.height, step_r):
            for c in range(0, self.width, step_c):
                self._start_list.append([c, r])

    def __preProcessing(self, im: np.ndarray) -> np.ndarray:
        im = im.transpose((1, 2, 0))
        im = im.astype("float32", copy=False) / 255.
        im -= self.mean
        im /= self.std
        return im

    def __getBlock(self, start_loc: List[int]) -> np.ndarray:
        xoff, yoff = start_loc
        xsize, ysize = self.block_size
        if xoff + xsize > self.width:
            xsize = self.width - xoff
        if yoff + ysize > self.height:
            ysize = self.height - yoff
        im = self._src_data.ReadAsArray(int(xoff), int(yoff), int(xsize), int(ysize))
        im = self.__preProcessing(im)
        h, w = im.shape[:2]
        out = np.zeros((self.block_size[1], self.block_size[0], 3), dtype=im.dtype)
        out[:h, :w, :] = im
        return {"block": out, "start": start_loc}