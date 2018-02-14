from keras_text_to_image.library.dcgan import DCGan
from keras_text_to_image.library.utility.img_cap_loader import load_img_cap
import numpy as np
from random import shuffle


def main():
    seed = 42

    np.random.seed(seed)

    img_dir_path = './data/pokemon/img'
    txt_dir_path = './data/pokemon/txt'
    model_dir_path = './models'

    img_width = 32
    img_height = 32

    image_label_pairs = load_img_cap(img_dir_path, txt_dir_path, img_width=img_width, img_height=img_height)

    shuffle(image_label_pairs)

    gan = DCGan()
    gan.load_model(model_dir_path)

    for i in range(3):
        image_label_pair = image_label_pairs[i]
        image = image_label_pair[0]
        text = image_label_pair[1]

        image.save('./data/outputs/' + DCGan.model_name + '-generated-' + str(i) + '-0.png')
        for j in range(3):
            generated_image = gan.generate_image_from_text(text)
            generated_image.save('./data/outputs/' + DCGan.model_name + '-generated-' + str(i) + '-' + str(j) + '.png')


if __name__ == '__main__':
    main()
