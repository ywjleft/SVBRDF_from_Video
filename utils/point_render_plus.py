import cv2, glob, os
import numpy as np
import tensorflow as tf

from utils.renderer import GGXRenderer
from utils.helpers import generateSurfaceArray

class Point_Render:
    def __init__(self, res):
        self.res = res
        surface_array = generateSurfaceArray(res)

        self.svbrdf = tf.placeholder(tf.float32, [1,res,res,12])
        self.lightpos = tf.placeholder(tf.float32, [3])
        self.lightdest = tf.placeholder(tf.float32, [3])
        self.viewpos = tf.placeholder(tf.float32, [3])

        wi = tf.reshape(self.lightpos - surface_array, (1,1,1,res,res,3))
        wo = tf.reshape(self.viewpos - surface_array, (1,1,1,res,res,3))

        rendererInstance = GGXRenderer()
        renderings = rendererInstance.render(self.svbrdf, wi, wo, self.lightdest, self.lightpos, [17.8, 17.8, 17.8, np.exp(1.6)])[0]
        renderings = tf.clip_by_value(renderings, 0.0, 1.0)
        renderings = tf.pow(renderings, 0.4545)
        self.renderings = tf.image.convert_image_dtype(renderings, dtype=tf.uint8, saturate=True)


    def render_images(self, svbrdf, lightposes, lightdests, viewposes, sess):
        ns = np.shape(lightposes)[0]
        rendered = np.zeros((ns, self.res, self.res, 3), np.uint8)
        for i in range(ns):
            rendered[i] = sess.run(self.renderings, feed_dict={self.svbrdf:svbrdf, self.lightpos:lightposes[i], self.lightdest:lightdests[i], self.viewpos:viewposes[i]})
        return rendered
