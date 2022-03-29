import codecs
import geojson
from geojson import Polygon, Feature, FeatureCollection
from typing import List, Dict
from .base import BaseWriter


class DetWriter(BaseWriter):
    def __init__(self, config: Dict) -> None:
        super(DetWriter, self).__init__(config, "json")
        self.feats = []
        self.rects = []
        self.draw_threshold = 0.5
        self.iou_threshold = 0.8
        self.dst_ds = codecs.open(self.save_path, "w", encoding="utf-8")

    def __gtConvert(self, x, y):
        gt = self.geotf
        x_geo = gt[0] + x * gt[1] + y * gt[2]
        y_geo = gt[3] + x * gt[4] + y * gt[5]
        return x_geo, y_geo

    def __calcIOU(self, box1, box2):
        xa = max(box1[0], box2[0])
        ya = max(box1[1], box2[1])
        xb = min(box1[2], box2[2])
        yb = min(box1[3], box2[3])
        inter_area = max(0, xb - xa + 1) * max(0, yb - ya + 1)
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou

    def __delReBox(self):
        pass
        # TODO: remove same

    def write(self, bboxs: Dict, start: List[int]) -> None:
        w, h = start
        for i in range(bboxs.shape[0]):
            clas, score, x1, y1, x2, y2 = bboxs[i, :].tolist()
            x1 += w
            y1 += h
            x2 += w
            y2 += h
            xg1, yg1 = self.__gtConvert(x1, y1)
            xg2, yg2 = self.__gtConvert(x2, y2)
            if score >= self.draw_threshold:
                poly = Polygon([[(xg1, yg1), (xg1, yg2), (xg2, yg2), (xg2, yg1), (xg1, yg1)]])
                feat = Feature(geometry=poly, properties={"class": int(clas), "score": score})
                self.rects.append((x1, y1, x2, y2))
                self.feats.append(feat)

    def close(self) -> None:
        self.__delReBox()
        gjs = FeatureCollection(self.feats)
        self.dst_ds.write(geojson.dumps(gjs))
        self.dst_ds.close()