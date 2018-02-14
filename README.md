# keras-text-to-image

Translate text to image in Keras using GAN and Word2Vec as well as recurrent neural networks


# Usage

The sample codes below only generate very small images, but the image size can be increased if you have sufficient
memory 

### Text-to-Image using GloVe and Deep Convolution GAN

Below is the [sample codes](demo/dcgan_train.py) to train the DCGan on a set of pokemon samples of pair (image, text)

```python
from keras_text_to_image.library.dcgan import DCGan
from keras_text_to_image.library.utility.img_cap_loader import load_normalized_img_and_its_text
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
    img_channels = 3

    image_label_pairs = load_normalized_img_and_its_text(img_dir_path, txt_dir_path, img_width=img_width, img_height=img_height)

    shuffle(image_label_pairs)

    gan = DCGan()
    gan.img_width = img_width
    gan.img_height = img_height
    gan.img_channels = img_channels
    gan.random_input_dim = 200
    gan.glove_source_dir_path = './very_large_data'

    batch_size = 16
    epochs = 1000
    gan.fit(model_dir_path=model_dir_path, image_label_pairs=image_label_pairs,
            snapshot_dir_path='./data/snapshots',
            snapshot_interval=100,
            batch_size=batch_size,
            epochs=epochs)


if __name__ == '__main__':
    main()

```

Below is the [sample codes](demo/dcgan_generate.py) on how to load the trained DCGan model to generate
3 new pokemon samples from each text description of a pokemon:

```python
from keras_text_to_image.library.dcgan import DCGan
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

    img_width = 32
    img_height = 32

    image_label_pairs = load_normalized_img_and_its_text(img_dir_path, txt_dir_path, img_width=img_width, img_height=img_height)

    shuffle(image_label_pairs)

    gan = DCGan()
    gan.load_model(model_dir_path)

    for i in range(3):
        image_label_pair = image_label_pairs[i]
        normalized_image = image_label_pair[0]
        text = image_label_pair[1]

        image = img_from_normalized_img(normalized_image)
        image.save('./data/outputs/' + DCGan.model_name + '-generated-' + str(i) + '-0.png')
        for j in range(3):
            generated_image = gan.generate_image_from_text(text)
            generated_image.save('./data/outputs/' + DCGan.model_name + '-generated-' + str(i) + '-' + str(j) + '.png')


if __name__ == '__main__':
    main()

```

# Configure to run on GPU on Windows

* Step 1: Change tensorflow to tensorflow-gpu in requirements.txt and install tensorflow-gpu
* Step 2: Download and install the [CUDA® Toolkit 9.0](https://developer.nvidia.com/cuda-90-download-archive) (Please note that
currently CUDA® Toolkit 9.1 is not yet supported by tensorflow, therefore you should download CUDA® Toolkit 9.0)
* Step 3: Download and unzip the [cuDNN 7.4 for CUDA@ Toolkit 9.0](https://developer.nvidia.com/cudnn) and add the
bin folder of the unzipped directory to the $PATH of your Windows environment 
