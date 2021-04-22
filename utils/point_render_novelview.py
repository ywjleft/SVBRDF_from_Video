import cv2, glob, os
import numpy as np
import tensorflow as tf

from utils.renderer import GGXRenderer
from utils.helpers import generateSurfaceArray

class Point_Render:
    def __init__(self, res):
        self.res = res
        surface_array = generateSurfaceArray(res)

        self.fov = tf.placeholder(tf.float32, [])
        self.svbrdf = tf.placeholder(tf.float32, [1,res,res,12])
        self.light_souc = tf.placeholder(tf.float32, [3])
        self.light_dest = tf.placeholder(tf.float32, [3])

        wi = tf.reshape(self.light_souc - surface_array, (1,1,1,res,res,3))
        light_souc = tf.reshape(self.light_souc, (1,1,1,1,1,3))
        light_dest = tf.reshape(self.light_dest, (1,1,1,1,1,3))

        rendererInstance = GGXRenderer()
        renderings = rendererInstance.render(self.svbrdf, wi, wi, light_dest, light_souc, [17.8, 17.8, 17.8, np.exp(1.6)])[0]
        renderings = tf.clip_by_value(renderings, 0.0, 1.0)
        renderings = tf.pow(renderings, 0.4545)
        self.renderings = tf.image.convert_image_dtype(renderings, dtype=tf.uint8, saturate=True)


    def render_image(self, svbrdf, lookats, fov, sess):
        cameraDist = 1.0 / (np.tan(fov / 2.0))
        renders = np.zeros((1+len(lookats), self.res, self.res, 3), np.uint8)
        renders[0] = sess.run(self.renderings, feed_dict={self.fov:fov, self.light_souc:[0,0,cameraDist], self.light_dest:[0,0,0], self.svbrdf:svbrdf})
        for i in range(len(lookats)):
            renders[i+1] = sess.run(self.renderings, feed_dict={self.fov:fov, self.light_souc:lookats[i,0], self.light_dest:lookats[i,0]+lookats[i,1], self.svbrdf:svbrdf})
        return renders
