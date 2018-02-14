from setuptools import find_packages
from setuptools import setup


setup(name='keras_gan_models',
      version='0.0.1',
      description='Generative Adversarial Network Models in Keras',
      author='Xianshun Chen',
      author_email='xs0040@gmail.com',
      url='https://github.com/chen0040/keras-gan-models',
      download_url='https://github.com/chen0040/keras-gan-models/tarball/0.0.1',
      license='MIT',
      install_requires=['Keras'],
      packages=find_packages())
