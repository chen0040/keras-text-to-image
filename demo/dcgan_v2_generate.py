from keras_text_to_image.library.dcgan_v2 import DCGanV2
from keras_text_to_image.library.utility.image_utils import img_from_normalized_img
from keras_text_to_image.library.utility.img_cap_loader import load_normalized_img_and_its_text
import numpy as np
from random import shuffle


def main():
    seed = 42

    np.random.seed(seed)

    img_dir_path = './data/pokemon/img'
    txt_dir_path = './data/pokemon/txt'
    model_dir_path = './models'

    img_width = 64
    img_height = 64

    image_label_pairs = load_normalized_img_and_its_text(img_dir_path, txt_dir_path, img_width=img_width, img_height=img_height)

    shuffle(image_label_pairs)

    gan = DCGanV2()
    gan.load_model(model_dir_path)

    for i in range(10):
        image_label_pair = image_label_pairs[i]
        normalized_image = image_label_pair[0]
        text = image_label_pair[1]

        image = img_from_normalized_img(normalized_image)
        image.save('./data/outputs/' + DCGanV2.model_name + '-generated-' + str(i) + '-0.png')
        for j in range(1):
            generated_image = gan.generate_image_from_text(text)
            generated_image.save('./data/outputs/' + DCGanV2.model_name + '-generated-' + str(i) + '-' + str(j) + '.png')


if __name__ == '__main__':
    main()
