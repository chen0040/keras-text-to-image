import os
import numpy as np

from keras.preprocessing.image import img_to_array, load_img


def load_and_scale_images(img_dir_path, extension, img_width, img_height):
    images = []
    for f in os.listdir(img_dir_path):
        filepath = os.path.join(img_dir_path, f)
        if os.path.isfile(filepath) and f.endswith(extension):
            image = img_to_array(load_img(filepath, target_size=(img_width, img_height)))
            image = (image.astype(np.float32) / 255) * 2 - 1
            images.append(image)
    return np.array(images)