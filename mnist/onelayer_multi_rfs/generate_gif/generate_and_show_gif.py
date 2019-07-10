# -*- coding: utf-8 -*-

import glob
import imageio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def generate_image_at_same_step():
    for step in range(1, 51):
        image_name = './images_at_epoch_{:04d}.png'.format(step)
        filenames = ['../rf={}/image_at_epoch_{:04d}.png'.format(rf_size, step) for rf_size in [1, 3, 5, 7, 9, 11, 13, 28]]
        images = [np.asarray(Image.open(filename)) for filename in filenames]
        plt.imsave(image_name, np.hstack(images))

def generate_gif(gif_name, image_pattern):
    with imageio.get_writer(gif_name, mode='I') as writer:
        filenames = glob.glob(image_pattern)
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame = 2 * (i**.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)

# rf_sizes = [1, 3, 5, 7, 9, 11, 13]
# for step in range(1, 51):
#     image_pattern = '../rf=*/image_at_epoch_{:04d}.png'.format(step)
#     
#     generate_image_at_same_step(image_pattern, image_name)

generate_image_at_same_step()
generate_gif('mnist.gif', 'images_at_epoch_*.png')