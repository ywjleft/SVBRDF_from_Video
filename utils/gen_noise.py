from noise import pnoise2
import numpy as np
import cv2

np.random.seed(111)
perlin_dataset = np.zeros((100,4096,4096), bool)
for i in range(100):
    octaves = np.random.randint(3) + 1
    freq = np.random.rand() * 48 + 16
    offsetx = np.random.rand() * 1024
    offsety = np.random.rand() * 1024
    threshold = np.random.rand() * 0.6 - 0.3
    for y in range(4096):
        for x in range(4096):
            perlin_dataset[i,y,x] = pnoise2(x / freq + offsetx, y / freq + offsety, octaves) > threshold

    if i < 10:
        cv2.imwrite('/path-to-write-training-data/PerlinNoise/{}.png'.format(i), perlin_dataset[i].astype(int)*200)

np.save('/path-to-write-training-data/PerlinNoise/Perlin4k.npy', perlin_dataset)