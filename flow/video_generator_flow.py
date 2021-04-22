import cv2, glob, os
import numpy as np
import tensorflow as tf

from utils.renderer import GGXRenderer
from utils.helpers import addNoise
from utils.warp_nearest import _interpolate_bilinear
from utils.utils import generate_normalized_random_direction

class Video_Generator:
    def __init__(self, res):
        self.res = res
        uvgrid = tf.constant(np.reshape(np.mgrid[0:res, 0:res], (2,res*res)), tf.float32)
        fov = tf.Variable(np.float32(0.0))
        self.ph_fov = tf.placeholder(tf.float32, [])
        self.assigner_fov = tf.assign(fov, self.ph_fov)
        f = 0.5 * res / (tf.math.tan(fov / 2.0))
        c = tf.constant(0.5 * res, tf.float32)
        svbrdf = tf.Variable(np.zeros([4096,4096,12], np.float32))
        self.ph_svbrdf = tf.placeholder(tf.float32, [4096,4096,12])
        self.assigner_svbrdf = tf.assign(svbrdf, self.ph_svbrdf)
        self.camera_param = tf.placeholder(tf.float32, [3,3])
        lightfactor = tf.Variable(np.zeros([4], np.float32))
        self.ph_lightfactor = tf.placeholder(tf.float32, [4])
        self.assigner_lightfactor = tf.assign(lightfactor, self.ph_lightfactor)

        extrinsic = self.lookAt(self.camera_param)
        a11 = (uvgrid[1] - c) * extrinsic[2,0] - f * extrinsic[0,0]
        a12 = (uvgrid[1] - c) * extrinsic[2,1] - f * extrinsic[0,1]
        a21 = (uvgrid[0] - c) * extrinsic[2,0] - f * extrinsic[1,0]
        a22 = (uvgrid[0] - c) * extrinsic[2,1] - f * extrinsic[1,1]
        b1 = f * extrinsic[0,3] + (c - uvgrid[1]) * extrinsic[2,3]
        b2 = f * extrinsic[1,3] + (c - uvgrid[0]) * extrinsic[2,3]

        detA = a11 * a22 - a21 * a12
        detX = b1 * a22 - b2 * a12
        detY = a11 * b2 - a21 * b1
        XY = tf.stack([detX / detA, detY / detA], axis=-1)
        resample_array = tf.reshape(XY, (res,res,2))

        resample_list = (1 + tf.reshape(resample_array, (1,res*res,2))) * 2048
        self.resampled_svbrdf = tf.reshape(_interpolate_bilinear(tf.expand_dims(svbrdf, axis=0), resample_list, indexing='xy'), 
            (1,res,res,12))

        surface_array = tf.concat([resample_array, tf.zeros((res,res,1), tf.float32)], axis=2)
        wi = tf.reshape(self.camera_param[0] - surface_array, (1,1,1,res,res,3))
        light_souc = tf.reshape(self.camera_param[0], (1,1,1,1,1,3))
        light_dest = tf.reshape(self.camera_param[1], (1,1,1,1,1,3))
        rendererInstance = GGXRenderer()
        renderings = rendererInstance.render(self.resampled_svbrdf, wi, wi, light_dest, light_souc, lightfactor)[0]

        ambientfactor = tf.Variable(np.zeros([4], np.float32))
        self.ph_ambientfactor = tf.placeholder(tf.float32, [4])
        self.assigner_ambientfactor = tf.assign(ambientfactor, self.ph_ambientfactor)
        wi_ambient = tf.reshape(ambientfactor[0:3] - surface_array, (1,1,1,res,res,3))
        renderings_ambient = rendererInstance.render_ambient(self.resampled_svbrdf, wi_ambient, wi, ambientfactor[3])[0]

        finalrender = addNoise(renderings + renderings_ambient)
        finalrender = tf.clip_by_value(finalrender, 0.0, 1.0)
        finalrender = tf.pow(finalrender, 0.4545)
        self.finalrender = tf.image.convert_image_dtype(finalrender, dtype=tf.uint8, saturate=True)

    def lookAt(self, camera_param):
        zp = camera_param[0] - camera_param[1]
        zp = zp / tf.norm(zp)
        
        xp = tf.linalg.cross(zp, camera_param[2])
        xp = xp / tf.norm(xp)

        yp = tf.linalg.cross(zp, xp)
        yp = yp / tf.norm(yp)

        out = tf.stack([tf.concat([xp, [-tf.tensordot(xp, camera_param[0], 1)]], 0), 
            tf.concat([yp, [-tf.tensordot(yp, camera_param[0], 1)]], 0), 
            tf.concat([zp, [-tf.tensordot(zp, camera_param[0], 1)]], 0)], 0)
        return out

    def render_video(self, svbrdf, fov, camera_params, sess, output_svbrdf_len=20): #4096*4096*10, 1, n*3*3
        lf = [0.1, 0.1, 0.1, np.exp(np.random.normal(1.6, 0.35))]
        ambientDir = generate_normalized_random_direction(0.001, 0.2)
        ambientPos = ambientDir * np.exp(np.random.normal(np.log(2.25), 0.2))
        af = np.concatenate([ambientPos, [np.exp(np.random.normal(np.log(0.1), 0.2))]])

        sess.run([self.assigner_fov, self.assigner_svbrdf, self.assigner_lightfactor, self.assigner_ambientfactor], 
            feed_dict={self.ph_fov:fov, self.ph_svbrdf:svbrdf, self.ph_lightfactor:lf, self.ph_ambientfactor:af})

        video = np.zeros((len(camera_params), self.res, self.res, 3), np.uint8)
        gtsvbrdf = np.zeros((output_svbrdf_len, self.res, self.res, 12), np.float32)

        for i in range(len(camera_params)):
            rendered, rs_svbrdf = sess.run([self.finalrender, self.resampled_svbrdf], 
                feed_dict={self.camera_param:camera_params[i]})
            video[i] = rendered[0]
            if i < output_svbrdf_len:
                gtsvbrdf[i] = rs_svbrdf[0]

        return video, gtsvbrdf

    def load_svbrdf(self, file, augment = True):
        svbrdf = np.load(file)
        if augment:
            if np.random.randint(2):
                svbrdf = svbrdf[::-1]
                svbrdf[...,1] = 255 - svbrdf[...,1]
            if np.random.randint(2):
                svbrdf = svbrdf[:,::-1]
                svbrdf[...,0] = 255 - svbrdf[...,0]
            if np.random.randint(2):
                svbrdf = np.transpose(svbrdf, (1,0,2))
                svbrdf[...,(0,1)] = -svbrdf[...,(1,0)]

        svbrdf = svbrdf / 255

        if np.shape(svbrdf)[1] == 4096:
            svbrdf[...,6:9] = svbrdf[...,6:9] ** 2

        if augment:
            maxv = np.amax(svbrdf, axis=(0,1))
            almul = np.random.rand() * (np.minimum(1/(np.amax(maxv[3:6])+0.001),1.25) - 0.8) + 0.8
            spmul = np.random.rand() * (np.minimum(1/(np.amax(maxv[9:12])+0.001),1.25) - 0.8) + 0.8
            romul = np.random.rand() * (np.minimum(1/(np.amax(maxv[6:9])+0.001),1.25) - 0.8) + 0.8
            mul = np.concatenate([[1,1,1], [almul, almul, almul], [romul, romul, romul], [spmul, spmul, spmul]])
            svbrdf = svbrdf * mul

        svbrdf[...,6:9] = np.maximum(svbrdf[...,6:9], 0.1)

        if np.shape(svbrdf)[1] == 1024:
            svbrdf = np.tile(svbrdf, [4,4,1])
        svbrdf = svbrdf * 2 - 1
        return svbrdf
        # normal diffuse roughness specular

    def generate_camera_params(self, length):
        camera_params = np.zeros((length,3,3), np.float32)

        size = np.random.rand() * 0.05 + 0.05
        fovRadian = (25.0 + np.random.rand() * 10.0) / 180.0 * np.pi
        cameraDist = size  / (np.tan(fovRadian / 2.0))
        scale = (0.8 + np.random.rand() * 0.4) * size / 12

        xylist = np.array([[0,0],[0,1],[0,2],[0,3],[0,4],[1,4],[2,4],[3,4],[4,4],
                    [4,3],[4,2],[4,1],[4,0],[4,-1],[4,-2],[4,-3],[4,-4],
                    [3,-4],[2,-4],[1,-4],[0,-4],[-1,-4],[-2,-4],[-3,-4],[-4,-4],
                    [-4,-3],[-4,-2],[-4,-1],[-4,0],[-4,1],[-4,2],[-4,3],[-4,4],[-4,5],[-4,6],[-4,7],[-4,8],
                    [-3,8],[-2,8],[-1,8],[0,8],[1,8],[2,8],[3,8],[4,8],[5,8],[6,8],[7,8],[8,8],
                    [8,7],[8,6],[8,5],[8,4],[8,3],[8,2],[8,1],[8,0],[8,-1],[8,-2],[8,-3],[8,-4],[8,-5],[8,-6],[8,-7],[8,-8],
                    [7,-8],[6,-8],[5,-8],[4,-8],[3,-8],[2,-8],[1,-8],[0,-8],[-1,-8],[-2,-8],[-3,-8],[-4,-8],[-5,-8],[-6,-8],[-7,-8],[-8,-8],
                    [-8,-7],[-8,-6],[-8,-5],[-8,-4],[-8,-3],[-8,-2],[-8,-1],[-8,0],[-8,1],[-8,2],[-8,3],[-8,4],[-8,5],[-8,6],[-8,7],[-8,8],[-8,9],[-8,10],[-8,11],[-8,12],
                    [-7,12],[-6,12],[-5,12],[-4,12],[-3,12],[-2,12],[-1,12],[0,12],[1,12],[2,12],[3,12],[4,12],[5,12],[6,12],[7,12],[8,12],[9,12],[10,12],[11,12],[12,12],
                    [12,11],[12,10],[12,9],[12,8],[12,7],[12,6],[12,5],[12,4],[12,3],[12,2],[12,1],[12,0],[12,-1],[12,-2],[12,-3],[12,-4],[12,-5],[12,-6],[12,-7],[12,-8]])

        if np.random.randint(2) == 1:
            xylist = xylist[:,::-1]
        if np.random.randint(2) == 1:
            xylist[:,0] = -xylist[:,0]
        if np.random.randint(2) == 1:
            xylist[:,1] = -xylist[:,1]

        x0 = np.random.rand() - 0.5
        y0 = np.random.rand() - 0.5
        condition = np.zeros(12) #vx,dx,vy,dy,vz,dz,vlx,dlx,vly,dly,vu,du

        for k in range(length):
            ax = np.random.randn() * 0.001 - condition[0] * 0.1 - condition[1] * 0.2
            condition[0] += ax
            condition[1] += condition[0]
            ay = np.random.randn() * 0.001 - condition[2] * 0.1 - condition[3] * 0.2
            condition[2] += ay
            condition[3] += condition[2]
            az = np.random.randn() * 0.001 - condition[4] * 0.01 - condition[5] * 0.1
            condition[4] += az
            condition[5] += condition[4]
            alx = np.random.randn() * 0.001 - condition[6] * 0.01 - condition[7] * 0.1
            condition[6] += alx
            condition[7] += condition[6]
            aly = np.random.randn() * 0.001 - condition[8] * 0.01 - condition[9] * 0.1
            condition[8] += aly
            condition[9] += condition[8]
            au = np.random.randn() * 0.001 - condition[10] * 0.01 - condition[11] * 0.1
            condition[10] += au
            condition[11] += condition[10]

            xf = np.clip(x0 + xylist[k,0] * scale + condition[1], -0.9, 0.9)
            yf = np.clip(y0 + xylist[k,1] * scale + condition[3], -0.9, 0.9)
            zf = cameraDist + condition[5]
            lxf = xf + condition[7]
            lyf = yf + condition[9]
            uf = condition[11]

            camera_params[k] = [[xf, yf, zf], [lxf, lyf, 0], [uf, 1, 0]]

        return camera_params, fovRadian

    def generate_video(self, svbrdf_file, video_length, svbrdf_length, sess):
        svbrdffull = self.load_svbrdf(svbrdf_file)
        camera_params, fovRadian = self.generate_camera_params(video_length)
        video, gtsvbrdf = self.render_video(svbrdffull, fovRadian, camera_params, sess, svbrdf_length)
        return video, gtsvbrdf, camera_params, fovRadian

    def save_sample_video(self, file, video):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(file, fourcc, 30.0, np.shape(video)[1:3])
        for k in range(np.shape(video)[0]):
            writer.write(video[k][...,::-1])
        writer.release()