import codecs
import itertools
import geojson
from geojson import Polygon, Feature, FeatureCollection
from typing import List, Dict
from .base import BaseWriter


class DetWriter(BaseWriter):
    def __init__(self, config: Dict) -> None:
        super(DetWriter, self).__init__(config, "json")
        self.rects = []
        self.draw_threshold = 0.5
        self.iou_threshold = 0.25
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
        dels = []
        temp = []
        for boxs in itertools.combinations(self.rects, 2):
            if self.__calcIOU(boxs[0][2:], boxs[1][2:]) > self.iou_threshold and \
               boxs[0][0] == boxs[1][0]:
                if boxs[0][1] > boxs[1][1]:
                    dels.append(boxs[1])
                else:
                    dels.append(boxs[0])
        dels = list(set(dels))
        for rect in self.rects:
            if rect not in dels:
                temp.append(rect)
        self.rects = temp

    def write(self, bboxs: Dict, start: List[int]) -> None:
        w, h = start
        for i in range(bboxs.shape[0]):
            clas, score, x1, y1, x2, y2 = bboxs[i, :].tolist()
            x1 += w
            y1 += h
            x2 += w
            y2 += h
            if score >= self.draw_threshold:
                self.rects.append((clas, score, x1, y1, x2, y2))   

    def __geoCoordinate(self) -> None:
        feats = []
        for rect in self.rects:
            clas, score, x1, y1, x2, y2 = rect
            xg1, yg1 = self.__gtConvert(x1, y1)
            xg2, yg2 = self.__gtConvert(x2, y2)
            poly = Polygon([[(xg1, yg1), (xg1, yg2), (xg2, yg2), (xg2, yg1), (xg1, yg1)]])
            feat = Feature(geometry=poly, properties={"class": int(clas), "score": score})
            feats.append(feat)
        return feats

    def close(self) -> None:
        self.__delReBox()
        feats = self.__geoCoordinate()
        gjs = FeatureCollection(feats)
        self.dst_ds.write(geojson.dumps(gjs))
        self.dst_ds.close()