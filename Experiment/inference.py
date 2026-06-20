import argparse
import re
import os

import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from CharRec.utils import CTCLabelConverter
from CharRec.model import Model

device = torch.device('cpu')


class ResizeNormalize:
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.to_tensor(img)
        img.sub_(0.5).div_(0.5)
        return img
 

def load_det(opt):
    detector = YOLO(opt.detector_path)
    return detector


def load_rec(opt):
    converter = CTCLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    recognizer = torch.nn.DataParallel(Model(opt)).to(device)
    recognizer.load_state_dict(torch.load(opt.recognizer_path, map_location=device))
    return recognizer
 
    
def PostProcessing(pred, mode="normal"):
    if pred is None:
        return pred
    pred = pred.strip().upper()
    
    BIC_PATTERN = re.compile(r'^[A-Z]{4}[0-9]{7}$')
    if BIC_PATTERN.fullmatch(pred):
        return pred
    
    if mode == "normal":
        if len(pred) == 10:
            if all(c.isalpha() for c in pred[:3]) and all(c.isdigit() for c in pred[3:]):
                pred = pred[:3] + 'U' + pred[3:]
                if BIC_PATTERN.fullmatch(pred):
                    return pred
    
    chars = list(pred)    
    CONFUSION = {'0': 'O',
                 '1': 'I',
                 '5': 'S',
                 '2': 'Z',
                 '8': 'B'}
    CONFUSION_R = {v: k for k, v in CONFUSION.items()}
    for i, ch in enumerate(chars):
        if i < 4:
            chars[i] = CONFUSION.get(ch, ch)
        else:
            chars[i] = CONFUSION_R.get(ch, ch)
            
    return ''.join(chars)    


class Inference:
    def __init__(self, opt, padding_size=4, postprocessing=True):
        self.opt = opt
        self.device = torch.device('cpu')
        self.padding_size = padding_size
        self.postprocessing = postprocessing
        self.converter = CTCLabelConverter(opt.character)
        self.opt.num_class = len(self.converter.character)
        self.transform = ResizeNormalize((150, 45))

    def detect(self, det, img):
        results = det.predict(img, verbose=False)
        if results[0].boxes.xyxy.shape[0] == 0:
            return None

        boxes = results[0].boxes
        best_idx = int(boxes.conf.argmax().item())
        xmin, ymin, xmax, ymax = boxes.xyxy[best_idx].tolist()
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])

        return xmin, ymin, xmax, ymax

    def recognize(self, rec, cropped_img):
        with torch.no_grad():
            preds = rec(self.transform(cropped_img.convert('L')).unsqueeze(0).to(self.device))
        preds_size = torch.IntTensor([preds.size(1)])
        _, preds_index = preds.max(2)
        
        pred_str = self.converter.decode(preds_index, preds_size)[0]
        if self.postprocessing:
            pred_str = PostProcessing(pred_str)

        return pred_str

    def inference(self, det, rec, img_path):
        det.eval()
        rec.eval()

        with Image.open(os.path.join(img_path)) as img:
            img = img.convert('RGB')
            img = img.resize((640, 640), Image.LANCZOS)

        bbox = self.detect(det, img)
        if bbox is None:
            print(f"Detection failed for image: {img_path}")
            return
        xmin, ymin, xmax, ymax = bbox
        cropped_img = img.crop((xmin - self.padding_size,
                                ymin - self.padding_size,
                                xmax + self.padding_size,
                                ymax + self.padding_size))

        preds_str = self.recognize(rec, cropped_img)
        print(f"Predicted result: {preds_str}")
        
        
def main():
    parser = argparse.ArgumentParser()

    """ Pipeline inputs """
    parser.add_argument("--image_path", required=True, help="path to test images")
    parser.add_argument("--detector_path", required=True, help="path to detector checkpoint")
    parser.add_argument("--recognizer_path", required=True, help="path to recognizer checkpoint")
    parser.add_argument("--padding", type=int, default=4)

    """ Recognizer data processing """
    parser.add_argument("--character", type=str, default="0123456789abcdefghijklmnopqrstuvwxyz")

    """ Recognizer model architecture """
    parser.add_argument("--FeatureExtraction", type=str, default='CNN_s')
    parser.add_argument("--input_channel", type=int, default=1)
    parser.add_argument("--output_channel", type=int, default=512)

    opt = parser.parse_args()
    det = load_det(opt)
    rec = load_rec(opt)

    measurer = Inference(opt=opt, padding_size=opt.padding)
    measurer.inference(det, rec, opt.image_path)        
    
if __name__ == "__main__":
    main()    