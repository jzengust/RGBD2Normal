from PIL import Image
import scipy.misc as m
import numpy as np

def png_reader_32bit(path, img_size=(0,0)):
    # 16-bit png will be read in as a 32-bit img
    image = Image.open(path)  
    pixel = np.array(image)
    if img_size[0]: #nearest interpolation
        step = pixel.shape[0]/img_size[0]
        pixel = pixel[0::step, :]
        pixel = pixel[:, 0::step]

    return pixel

def png_reader_uint8(path, img_size=(0,0)):
    image = Image.open(path)
    pixel = np.array(image, dtype=np.uint8)
    if img_size[0]:
        pixel = m.imresize(pixel, (img_size[0], img_size[1]))#only works for 8 bit image

    return pixel