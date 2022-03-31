import paddle
from paddleseg.models.segformer import SegFormer_B2
from ppdet.core.workspace import load_config, create
from paddleclas.ppcls.arch.backbone.model_zoo.ghostnet import GhostNet_x1_3
from sinfer import SegSlider, DetSlider, ClsSlider


if __name__ == "__main__":
    # Seg
    tif_path = "assets/image/test1.tif"
    params_path = "assets/model/segformer_b2_512x512_rs_building.pdparams"
    model = SegFormer_B2(num_classes=2)
    model.set_state_dict(paddle.load(params_path))
    slide_model = SegSlider(model)
    slide_model(tif_path)

    # Det
    tif_path = "assets/image/test2.tif"
    params_path = "assets/model/yolov3_mobilenet_v3_608x608_rs_car.pdparams"
    config_path = "assets/config/yolov3_mobilenet_v3_large.yml"
    cfg = load_config(config_path)
    model = create(cfg.architecture)
    model.set_state_dict(paddle.load(params_path))
    slide_model = DetSlider(model)
    slide_model(tif_path)

    # Cls
    tif_path = "assets/image/test3.tif"
    params_path = "assets/model/ghostnet_x1_128x128_rs_landuse.pdparams"
    model = GhostNet_x1_3(class_num=9)
    model.set_state_dict(paddle.load(params_path))
    slide_model = ClsSlider(model)
    slide_model(tif_path)