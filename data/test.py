from datasets import load_dataset
import webdataset as wds
from PIL import Image, ImageDraw
import numpy as np

ds = load_dataset("Miguel231/refcocog_polygons", split="train")

for key in ds:
    print(key)
    break
