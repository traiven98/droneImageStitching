#Author Kyryl Serediuk and Vladyslav Varfolomeev
from PIL import Image, ImageEnhance
import random
import numpy as np


def shadow_V(image_path, area = 0.3):
    im = Image.open(image_path)
    width, height = im.size
    crop_width = int(width * area)
    crop_height = int(height * area)

    x = random.randint(0, width - crop_width)
    y = random.randint(0, height - crop_height)

    im_crop = im.crop((x, y, x+crop_width, y+crop_height))

    enhancer = ImageEnhance.Brightness(im_crop)
    factor = 0.5

    im_output = enhancer.enhance(factor)

    im.paste(im_output, (x, y, x+crop_width, y+crop_height))
    open_cv_image = np.array(im)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image
