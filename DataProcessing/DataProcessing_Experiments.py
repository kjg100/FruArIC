import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def clean_data(exp_img, exp_json):
    exp_img = Path(exp_img)
    exp_json = Path(exp_json)

    jsons = sorted(exp_json.glob("*.json"))
    duplicated = set()

    for j in jsons:
        with open(j, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        annot = data.get("annotations", [])
        bic_annot = [a for a in annot if a.get("bbox", {}).get("classid") == "BIC"]
        if len(bic_annot) != 1:
            os.remove(j)
            continue

        bbox = bic_annot[0].get("bbox", {})
        text_val = str(bbox.get("text", "")).strip()
        if text_val == "":
            os.remove(j)
            continue

        if text_val in duplicated:
            os.remove(j)
            continue
        duplicated.add(text_val)

        data["annotations"] = bic_annot
        with open(j, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    stem_x = {p.stem for p in exp_json.glob("*.json")}
    for path_i in exp_img.glob("*.jpg"):
        if path_i.stem not in stem_x:
            os.remove(path_i)

    print("image:", len(os.listdir(exp_img)))
    print("label:", len(os.listdir(exp_json)))


def select_data(src_img, src_json, dst_img, dst_json, n_select):
    src_img = Path(src_img)
    src_json = Path(src_json)
    dst_img = Path(dst_img)
    dst_json = Path(dst_json)

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_json.mkdir(parents=True, exist_ok=True)

    img_lst = sorted([p.name for p in src_img.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    indices = np.linspace(0, len(img_lst) - 1, n_select, dtype=int)
    selected_lst = [img_lst[i] for i in indices]

    for img_name in selected_lst:
        img_src = src_img / img_name
        lab_src = src_json / Path(img_name).with_suffix(".json").name

        img_dst = dst_img / img_name
        lab_dst = dst_json / Path(img_name).with_suffix(".json").name

        shutil.move(str(img_src), str(img_dst))
        shutil.move(str(lab_src), str(lab_dst))

    print(f"Source total: {len(img_lst)}")
    print("Remaining source image:", len(os.listdir(src_img)))
    print("Remaining source label:", len(os.listdir(src_json)))

    print(f"Selected total: {len(selected_lst)}")
    print("Selected image:", len(os.listdir(dst_img)))
    print("Selected label:", len(os.listdir(dst_json)))


def build_data(exp_img, exp_json, processed_dir):
    exp_img = Path(exp_img)
    exp_json = Path(exp_json)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    for jp in tqdm(sorted(exp_json.glob("*.json")), desc="making processed experiment images"):
        w_target, h_target = 640, 640
        data = json.loads(jp.read_text(encoding="utf-8"))

        annotations = []
        for ann in data.get("annotations", []):
            bbox = ann.get("bbox", {})
            text_val = str(bbox.get("text", "")).strip()
            if text_val:
                annotations.append(ann)

        img = Image.open(f"{exp_img / jp.stem}.jpg").convert("RGB")
        img_resized = img.resize((w_target, h_target), Image.BILINEAR)

        for idx, ann in enumerate(annotations):
            bbox = ann.get("bbox", {})
            text_val = str(bbox.get("text", "")).strip()
            img_resized.save(f"{processed_dir / text_val}.jpg", quality=99)

    print("Processed image:", len(os.listdir(processed_dir)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_img", required=True, help="path to source experiment image directory")
    parser.add_argument("--src_json", required=True, help="path to source experiment json directory")
    parser.add_argument("--exp_img", required=True, help="path to selected experiment image directory")
    parser.add_argument("--exp_json", required=True, help="path to selected experiment json directory")
    parser.add_argument("--processed_dir", required=True, help="path to processed experiment image directory")
    parser.add_argument("--n_select", type=int, default=1000)

    opt = parser.parse_args()
    clean_data(opt.src_img, opt.src_json)
    select_data(opt.src_img, opt.src_json, opt.exp_img, opt.exp_json, opt.n_select)
    build_data(opt.exp_img, opt.exp_json, opt.processed_dir)
