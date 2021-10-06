import os
import argparse
import yaml
import mxnet as mx
import pandas as pd
import numpy as np
from skimage.filters import gaussian
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Dict, Optional, Tuple



def create_data_generator(config: Dict, shuffle: bool = False, save_img: bool=True):
    """
    Wrapper around 'flow_from_dataframe'-method. Uses loaded dataframes to load images and labels.
    """
    def hue_jitter(img):
        if config['augmentation']["hue"] > 0.0:
            aug = mx.image.HueJitterAug(hue=config['augmentation']["hue"])
            img = aug(img)
        return img

    def saturation_jitter(img):
        if config['augmentation']["saturation"] > 0.0:
            aug = mx.image.SaturationJitterAug(saturation=config['augmentation']["saturation"])
            img = aug(img)
        return img

    def contrast_jitter(img):
        if config['augmentation']["contrast"] > 0.0:
            aug = mx.image.ContrastJitterAug(contrast=config['augmentation']["contrast"])
            img = aug(img)
        return img

    def brightness_jitter(img):
        if config['augmentation']["contrast"] > 0.0:
            aug = mx.image.BrightnessJitterAug(brightness=config['augmentation']["brightness"])
            img = aug(img)
        return img

    def gaussian_blurr(img):
        if config['augmentation']["blur"] > 0.0:
            sigma = np.random.uniform(0, config['augmentation']["blur"], 1)
            img = gaussian(img, sigma=sigma[0], multichannel=True)
        return img

    def custom_augmentation(img):
        img = mx.nd.array(img)
        img = hue_jitter(img)
        img = saturation_jitter(img)
        img = contrast_jitter(img)
        img = brightness_jitter(img)
        img = img.asnumpy()
        img = gaussian_blurr(img)
        return img

    datagen = ImageDataGenerator(
        width_shift_range=config['augmentation']["width_shift_range"],
        height_shift_range=config['augmentation']["height_shift_range"],
        channel_shift_range=config['augmentation']["channel_shift_range"],
        zoom_range=config['augmentation']["zoom_range"],
        rotation_range=0,
        fill_mode='constant',
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=custom_augmentation)
    
    if save_img:
        save_to_dir = config['output_dir']
    else:
        save_to_dir = None
    
    
    generator = datagen.flow_from_directory(
        directory=config["input_dir"],
        batch_size=1,
        shuffle=shuffle,
        classes=None,
        class_mode=None,
        save_to_dir=save_to_dir,
        save_format='jpeg'
    )

    return generator

def main(config):
    os.makedirs(config['output_dir'], exist_ok=True)
    data_gen = create_data_generator(config)
    for i in range(data_gen.n):
        data_gen.next()
    print('Output examples saved to: ' + config['output_dir'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cancer Classification")
    parser.add_argument("--config", "-c", type=str, default="./config.yaml",
                        help="Config path (yaml file expected) to default config.")
    args = parser.parse_args()
    with open(args.config) as file:
        config = yaml.full_load(file)

    main(config)

