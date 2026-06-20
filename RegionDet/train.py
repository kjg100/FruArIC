import argparse
from ultralytics import RTDETR, YOLO


def build_model(model_cfg: str, model_type: str = "yolo"):
    model_type = model_type.lower()
    if model_type == "yolo":
        return YOLO(model_cfg)
    if model_type == "rtdetr":
        return RTDETR(model_cfg)


def train(opt):
    model = build_model(opt.model_cfg, opt.model_type)
    model.info()
    model.train(data=opt.data,
                epochs=opt.epochs,
                patience=opt.patience,
                imgsz=opt.imgsz,
                seed=opt.seed,
                name=opt.name if opt.name else f"{opt.exp_prefix}_seed{opt.seed}",)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_cfg", type=str, default=r".\cfg\TailoredDet.yaml", help="Path to model YAML file or checkpoint.")
    parser.add_argument("--model_type", type=str, default="yolo", choices=["yolo", "rtdetr"], help="Ultralytics model type.")
    parser.add_argument("--data", type=str, default=None,help="Path to dataset YAML file.",)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--exp_prefix", type=str, default="td11")
    
    opt = parser.parse_args()
    train(opt)
