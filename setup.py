from setuptools import find_packages
from setuptools import setup


setup(name='keras_text_to_image',
      version='0.0.1',
      description='Text to Image using GloVe and Word2Vec with Generative Adversarial Network Models in Keras',
      author='Xianshun Chen',
      author_email='xs0040@gmail.com',
      url='https://github.com/chen0040/keras-text-to-image',
      download_url='https://github.com/chen0040/keras-text-to-image/tarball/0.0.1',
      license='MIT',
      install_requires=['Keras'],
      packages=find_packages())
