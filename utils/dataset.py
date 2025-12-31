from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image, ImageDraw
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
import toml
from abc import ABC, abstractmethod
import ast  

from models.text_backbone import MultiModalTextEncoder 

class CommonDataset(ABC):
    def __init__(self, ds, config, transform = None):
        self.ds = ds
        self.config = config
        self.transform = transform

    @abstractmethod
    def __getitem__(self, id):
        raise NotImplementedError

    def __len__(self):
        return len(self.ds)

class Box2SegmentDataset(CommonDataset, Dataset):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    def __init__(self, ds, config, transform=None):
        super().__init__(ds, config, transform)
        #self.text_processor = MultiModalTextEncoder().to(self.device)

    @staticmethod
    def box_to_mask(shape, box):
        h, w = shape
        mask_img = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask_img)

        x0, y0, bw, bh = map(int, box)
        x1, y1 = x0 + bw, y0 + bh
        draw.rectangle([x0, y0, x1, y1], fill=1)

        return np.array(mask_img)

    def __getitem__(self, id):
        img = self.ds[id]["jpg"].convert("RGB")
        box = self.ds[id]["json"]
        txt = self.ds[id]["txt"]

        img = np.array(img)
        mask = self.box_to_mask(img.shape[:2], box)

        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        return img, mask, txt

    def show_sample(self, id, transformed=True):
        from matplotlib.colors import ListedColormap

        if transformed:
            img, mask, _ = self.__getitem__(id)
            img = img.permute(1, 2, 0).numpy()
            mask = mask.numpy()
        else:
            img = np.array(self.ds[id]["jpg"])
            box = self.ds[id]["json"]
            mask = self.box_to_mask(img.shape[:2], box)

        custom_cmap = ListedColormap([(0, 0, 0, 0), (1, 0.4, 0.4, 0.5)])

        plt.imshow(img)
        plt.imshow(mask, cmap=custom_cmap)
        plt.show()

class Poly2SegmentDataset(CommonDataset, Dataset):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __init__(self, ds, config, transform=None):
        super().__init__(ds, config, transform)
        # self.text_processor = MultiModalTextEncoder().to(self.device)

    @staticmethod
    def poly_to_mask(shape, poly_bytes):
        h, w = shape
        mask_img = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask_img)

        try:
            poly_str = poly_bytes.decode('utf-8')
            polygons = ast.literal_eval(poly_str)
            
            for points in polygons:
                draw.polygon(points, fill=1, outline=1)
                
        except Exception as e:
            print(f"Error parsing polygon: {e}")
            pass

        return np.array(mask_img)

    def __getitem__(self, id):
        img = self.ds[id]["jpg"].convert("RGB")
        
        poly_data = self.ds[id]["poly"]
        txt = self.ds[id]["txt"]

        img_np = np.array(img)
        mask = self.poly_to_mask(img_np.shape[:2], poly_data)

        if self.transform:
            transformed = self.transform(image=img_np, mask=mask)
            img_tensor = transformed['image']
            mask_tensor = transformed['mask']
        else:
            img_tensor = img_np
            mask_tensor = mask

        return img_tensor, mask_tensor, txt, id

    def show_sample(self, id, transformed=True):
        from matplotlib.colors import ListedColormap

        if transformed:
            img, mask, _ = self.__getitem__(id)

            if isinstance(img, torch.Tensor):
                img = img.permute(1, 2, 0).numpy()
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
        else:
            img = np.array(self.ds[id]["jpg"].convert("RGB"))
            poly_data = self.ds[id]["poly"]
            mask = self.poly_to_mask(img.shape[:2], poly_data)

        custom_cmap = ListedColormap([(0, 0, 0, 0), (1, 0.4, 0.4, 0.5)])

        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.imshow(mask, cmap=custom_cmap)
        plt.axis('off')
        plt.title(f"Sample {id}")
        plt.savefig("training_visualizations/sample.png")

    def save_prediction(self, pred_mask, id, save_path, thresh=0.5):
        import os
        from matplotlib.colors import ListedColormap

        img_tensor, gt_mask, txt, _ = self.__getitem__(id)
        
        if isinstance(img_tensor, torch.Tensor):
            img_disp = img_tensor.permute(1, 2, 0).cpu().numpy()

            mean = np.array([0.48145466, 0.4578275, 0.40821073])
            std = np.array([0.26862954, 0.26130258, 0.27577711])
            img_disp = (img_disp * std + mean)
            img_disp = np.clip(img_disp, 0, 1)

        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.detach().cpu().numpy()
        
        while pred_mask.ndim > 2:
            pred_mask = pred_mask.squeeze(0)

        binary_mask = (pred_mask > thresh).astype(float)

        print(f"[ID {id}] Max Conf: {pred_mask.max():.4f} | Mean Conf: {pred_mask.mean():.4f}")

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        ax[0].imshow(img_disp)
        ax[0].set_title(f"Input: {txt}")
        ax[0].axis('off')

        im = ax[1].imshow(pred_mask, vmin=0, vmax=1, cmap='jet')
        ax[1].set_title(f"Probability Map (Max: {pred_mask.max():.2f})")
        ax[1].axis('off')
        plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)

        custom_cmap = ListedColormap([(0, 0, 0, 0), (0, 1, 0, 0.5)])
        ax[2].imshow(img_disp)
        ax[2].imshow(binary_mask, cmap=custom_cmap, vmin=0, vmax=1)
        ax[2].set_title("Binary Pred (> 0.5)")
        ax[2].axis('off')

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    try:
        config = toml.load("configs/basic.toml")
    except FileNotFoundError:
        config = {
            "utils": {
                "dataset_name": "Miguel231/refcocog_polygons", 
                "target_size": [640, 640]
            }
        }

    ds = load_dataset("Miguel231/refcocog_polygons", split="train")

    target_size = config["utils"]["target_size"]

    transform = A.Compose([
        A.Resize(height=target_size[0], width=target_size[1]),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=35, p=1.0),
        ToTensorV2(),
    ])

    dataset = Poly2SegmentDataset(ds, config, transform=transform)

    if len(dataset) > 0:
        img, mask, txt = dataset[0]

        print(f"Image Shape: {img.shape}")
        print(f"Mask Shape: {mask.shape}")
        print(f"Mask Values: Min={mask.min()}, Max={mask.max()}")
        print(f"Text: {txt}")

        dataset.show_sample(4312, transformed=True)
        
#        print("Visualizing sample 0 (Original)...")
#        dataset.show_sample(0, transformed=False)
