import numpy as np
from PIL import Image

image_path = "input/train_hq/0cdf5b5d0ce1_01.jpg"
mask_path = "input/train_masks/0cdf5b5d0ce1_01_mask.gif"

image = np.array(Image.open(image_path))
mask = np.array(Image.open(mask_path))

print(image.shape)
print(mask.shape)
