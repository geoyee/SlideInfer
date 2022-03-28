# SlideInfer

用于简单包装PaddleSeg和PaddleDetection的模型，能够直接在训练完成的动态图模型上对遥感影像（来自地图下载器的RGB-UINT8-TIF）进行推理。

```python
import paddle
from paddleseg.models.segformer import SegFormer_B2
from sinfer import SegSlider

model = SegFormer_B2(num_classes=2)
model.set_dict(paddle.load(params_path))

# 转换为滑框推理模型
slide_model = SegSlider(model)
# # 可选，设置一些参数
# slide_model.ready(
#    block_size: 512,
#    overlap: 32,
#    mean: [0.5, 0.5, 0.5], 
#    std: [0.5, 0.5, 0.5]
# )
# 滑框推理
slide_model(tif_path)
```

## TODO

- [x] 语义分割（保存格式：geotiff）
- [ ] 矩形框识别（保存格式：geojson）
