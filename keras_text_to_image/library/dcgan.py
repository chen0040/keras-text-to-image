from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, concatenate
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras_text_to_image.library.utility.image_utils import combine_images
from keras import backend as K
import numpy as np
from PIL import Image
import os

from keras_text_to_image.library.utility.glove_loader import GloveModel


def generator_model(random_input_dim=None, text_input_dim=None, img_width=None, img_height=None, img_channels=None):
    if random_input_dim is None:
        random_input_dim = 100

    if text_input_dim is None:
        text_input_dim = 100

    if img_width is None:
        img_width = 7

    if img_height is None:
        img_height = 7

    if img_channels is None:
        img_channels = 1

    init_img_width = img_width // 4
    init_img_height = img_height // 4

    random_input = Input(shape=(random_input_dim, ))
    text_input = Input(shape=(text_input_dim, ))
    random_dense = Dense(1024)(random_input)
    text_dense = Dense(1024)(text_input)

    merged = concatenate([random_dense, text_dense])
    layer = Activation('tanh')(merged)

    layer = Dense(128 * init_img_width * init_img_height)(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('tanh')(layer)
    layer = Reshape((init_img_width, init_img_height, 128), input_shape=(128 * init_img_width * init_img_height,))(layer)
    layer = UpSampling2D(size=(2, 2))(layer)
    layer = Conv2D(64, kernel_size=5, padding='same')(layer)
    layer = Activation('tanh')(layer)
    layer = UpSampling2D(size=(2, 2))(layer)
    layer = Conv2D(img_channels, kernel_size=5, padding='same')(layer)
    output = Activation('tanh')(layer)

    model = Model([random_input, text_input], output)

    model.compile(loss='mean_squared_error', optimizer="SGD")

    print(model.summary())

    return model


def discriminator_model(img_width=None, img_height=None, img_channels=None, text_input_dim=None):
    if img_width is None:
        img_width = 28

    if img_height is None:
        img_height = 28

    if img_channels is None:
        img_channels = 1

    if text_input_dim is None:
        text_input_dim = 100

    text_input = Input(shape=(text_input_dim,))
    text_model = Dense(1024)(text_input)

    img_input = Input(shape=(img_width, img_height, img_channels))
    img_model = Conv2D(64, kernel_size=(5, 5), padding='same', input_shape=(img_width, img_height, img_channels))(img_input)
    img_model = Activation('tanh')(img_model)
    img_model = MaxPooling2D(pool_size=(2, 2))(img_model)
    img_model = Conv2D(128, kernel_size=5)(img_model)
    img_model = Activation('tanh')(img_model)
    img_model = MaxPooling2D(pool_size=(2, 2))(img_model)
    img_model = Flatten()(img_model)
    img_model = Dense(1024)(img_model)

    merged = concatenate([img_model, text_model])

    layer = Activation('tanh')(merged)
    layer = Dense(1)(layer)
    output = Activation('sigmoid')(layer)

    model = Model([img_input, text_input], output)

    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=d_optim)
    print(model.summary())
    return model


def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    model.compile(
        loss='binary_crossentropy', optimizer=g_optim)

    print(model.summary())
    return model


class DCGan(object):
    model_name = 'dc-gan'

    def __init__(self):
        K.set_image_dim_ordering('tf')
        self.generator = None
        self.discriminator = None
        self.model = None
        self.img_width = 7
        self.img_height = 7
        self.img_channels = 1
        self.random_input_dim = 100
        self.text_input_dim = 100
        self.config = None
        self.glove_source_dir_path = './very_large_data'
        self.glove_model = GloveModel()

    @staticmethod
    def get_config_file_path(model_dir_path):
        return os.path.join(model_dir_path, DCGan.model_name + '-config.npy')

    @staticmethod
    def get_weight_file_path(model_dir_path, model_type):
        return os.path.join(model_dir_path, DCGan.model_name + '-' + model_type + '-weights.h5')

    def create_model(self):
        self.generator = generator_model(self.random_input_dim, self.img_width, self.img_height, self.img_channels)
        self.discriminator = discriminator_model(self.img_width, self.img_height, self.img_channels)
        self.model = generator_containing_discriminator(self.generator, self.discriminator)

    def load_model(self, model_dir_path):
        config_file_path = DCGan.get_config_file_path(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.img_width = self.config['img_width']
        self.img_height = self.config['img_height']
        self.img_channels = self.config['img_channels']
        self.random_input_dim = self.config['random_input_dim']
        self.text_input_dim = self.config['text_input_dim']
        self.glove_source_dir_path = self.config['glove_source_dir_path']
        self.create_model()
        self.generator.load_weights(DCGan.get_weight_file_path(model_dir_path, 'generator'))
        self.discriminator.load_weights(DCGan.get_weight_file_path(model_dir_path, 'discriminator'))

    def fit(self, model_dir_path, image_label_pairs, epochs=None, batch_size=None, snapshot_dir_path=None, snapshot_interval=None):
        if epochs is None:
            epochs = 100

        if batch_size is None:
            batch_size = 128

        if snapshot_interval is None:
            snapshot_interval = 20

        self.config = dict()
        self.config['img_width'] = self.img_width
        self.config['img_height'] = self.img_height
        self.config['random_input_dim'] = self.random_input_dim
        self.config['text_input_dim'] = self.text_input_dim
        self.config['img_channels'] = self.img_channels
        self.config['glove_source_dir_path'] = self.glove_source_dir_path

        self.glove_model.load(data_dir_path=self.glove_source_dir_path, embedding_dim=self.text_input_dim)

        config_file_path = DCGan.get_config_file_path(model_dir_path)

        np.save(config_file_path, self.config)
        noise = np.zeros((batch_size, self.random_input_dim))

        self.create_model()

        for epoch in range(epochs):
            print("Epoch is", epoch)
            batch_count = int(image_label_pairs.shape[0] / batch_size)
            print("Number of batches", batch_count)
            for batch_index in range(batch_count):
                # Step 1: train the discriminator

                # initialize random input
                for i in range(batch_size):
                    noise[i, :] = np.random.uniform(-1, 1, self.random_input_dim)

                image_label_pair_batch = image_label_pairs[batch_index * batch_size:(batch_index + 1) * batch_size]

                # image_batch = np.transpose(image_batch, (0, 2, 3, 1))
                generated_images = self.generator.predict(noise, verbose=0)

                if (epoch * batch_size + batch_index) % snapshot_interval == 0 and snapshot_dir_path is not None:
                    self.save_snapshots(generated_images, snapshot_dir_path=snapshot_dir_path,
                                        epoch=epoch, batch_index=batch_index)

                correct_image_label_pair_batch = []
                generated_image_label_pair_batch = []
                for index in range(batch_size):
                    image_label_pair = image_label_pair_batch[index]
                    generated_image = generated_images[index]
                    img = image_label_pair[0]
                    text = image_label_pair[1]
                    encoded_text = self.glove_model.encode_doc(text)
                    generated_image_label_pair_batch.append([generated_image, encoded_text])
                    correct_image_label_pair_batch.append([img, encoded_text])

                correct_image_label_pair_batch = np.array(correct_image_label_pair_batch)
                generated_image_label_pair_batch = np.array(generated_image_label_pair_batch)

                X = np.concatenate((correct_image_label_pair_batch, generated_image_label_pair_batch))
                Y = np.array([1] * batch_size + [0] * batch_size)

                print('batch shape: ', X.shape)

                self.discriminator.trainable = True
                d_loss = self.discriminator.train_on_batch(X, Y)
                print("Epoch %d batch %d d_loss : %f" % (epoch, batch_index, d_loss))

                # Step 2: train the generator
                noise_image_label_pair_batch = []
                for index in range(batch_size):
                    image_label_pair = image_label_pair_batch[index]
                    text = image_label_pair[1]
                    encoded_text = self.glove_model.encode_doc(text)
                    noise[index, :] = np.random.uniform(-1, 1, self.random_input_dim)
                    noise_image_label_pair_batch.append([noise, encoded_text])
                noise_image_label_pair_batch = np.array(noise_image_label_pair_batch)
                self.discriminator.trainable = False
                g_loss = self.model.train_on_batch(noise_image_label_pair_batch, np.array([1] * batch_size))

                print("Epoch %d batch %d g_loss : %f" % (epoch, batch_index, g_loss))
                if (epoch * batch_size + batch_index) % 10 == 9:
                    self.generator.save_weights(DCGan.get_weight_file_path(model_dir_path, 'generator'), True)
                    self.discriminator.save_weights(DCGan.get_weight_file_path(model_dir_path, 'discriminator'), True)

        self.generator.save_weights(DCGan.get_weight_file_path(model_dir_path, 'generator'), True)
        self.discriminator.save_weights(DCGan.get_weight_file_path(model_dir_path, 'discriminator'), True)

    def generate_image(self):
        noise = np.zeros(shape=(1, self.random_input_dim))
        noise[0, :] = np.random.uniform(-1, 1, self.random_input_dim)
        generated_images = self.generator.predict(noise, verbose=0)
        generated_image = generated_images[0]
        generated_image = generated_image * 127.5 + 127.5
        return Image.fromarray(generated_image.astype(np.uint8))

    def save_snapshots(self, generated_images, snapshot_dir_path, epoch, batch_index):
        image = combine_images(generated_images)
        image = image * 127.5 + 127.5
        Image.fromarray(image.astype(np.uint8)).save(
            os.path.join(snapshot_dir_path, DCGan.model_name + '-' + str(epoch) + "-" + str(batch_index) + ".png"))
