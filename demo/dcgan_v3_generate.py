import os
import sys
import numpy as np
from random import shuffle


def main():
    seed = 42
    np.random.seed(seed)

    current_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(current_dir, '..'))

    img_dir_path = current_dir + '/data/pokemon/img'
    txt_dir_path = current_dir + '/data/pokemon/txt'
    model_dir_path = current_dir + '/models'

    img_width = 32
    img_height = 32

    from keras_text_to_image.library.dcgan_v3 import DCGanV3
    from keras_text_to_image.library.utility.image_utils import img_from_normalized_img
    from keras_text_to_image.library.utility.img_cap_loader import load_normalized_img_and_its_text

    image_label_pairs = load_normalized_img_and_its_text(img_dir_path, txt_dir_path, img_width=img_width, img_height=img_height)

    shuffle(image_label_pairs)

    gan = DCGanV3()
    gan.load_model(model_dir_path)

    for i in range(10):
        image_label_pair = image_label_pairs[i]
        normalized_image = image_label_pair[0]
        text = image_label_pair[1]

        image = img_from_normalized_img(normalized_image)
        image.save(current_dir + '/data/outputs/' + DCGanV3.model_name + '-generated-' + str(i) + '-0.png')
        for j in range(3):
            generated_image = gan.generate_image_from_text(text)
            generated_image.save(current_dir + '/data/outputs/' + DCGanV3.model_name + '-generated-' + str(i) + '-' + str(j) + '.png')


if __name__ == '__main__':
    main()
