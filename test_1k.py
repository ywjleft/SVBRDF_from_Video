# This test script reconstructs 1k-resolution SVBRDF maps. It takes the first frame as the guidance. 

import tensorflow as tf
import numpy as np
import argparse, os, cv2, glob
from ctypes import *

from flow.model_flow import PWCDCNet
from svbrdf.model_svbrdf_bigmap import Model
from utils.utils import warp_flow_bv, max_valid_noinit, upsample2, downsample, saveSVBRDF


parser = argparse.ArgumentParser()

'''
The input path can be a npy file, or an avi file, or a folder, and will be processed as follows:
npy file: loaded by numpy, it should be a lenx1024x1024x3 array
avi file: loaded by opencv, it should be a lenx1024x1024x3 video
folder: scanned by glob for npy and avi files
'''

parser.add_argument('-input', dest='input', default='./example_test_data/1k.npy', help='the input path')
parser.add_argument('-output', dest='output', default='./output', help='the folder to output svbrdf maps')
parser.add_argument('-gpuid', dest='gpuid', default='0', help='the value for CUDA_VISIBLE_DEVICES')

parser.add_argument('-model_adjacent', dest='model_adjacent', default='./trained_models/adjacent.ckpt')
parser.add_argument('-model_refinement', dest='model_refinement', default='./trained_models/distant.ckpt')
parser.add_argument('-model_svbrdf', dest='model_svbrdf', default='./trained_models/svbrdf.ckpt')


args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

os.makedirs(args.output, exist_ok=True)

bs = 1
res0 = 256
res = 1024
fmstep = 5

fov = 30 / 180 * np.pi
f = 0.5 * res0 / np.tan(fov/2)
c = 0.5 * res0
intrinsic = np.array([[f,0,c],[0,f,c],[0,0,1]])

path = './utils/combine_flow.so'
if not os.path.isfile(path):
    os.system('g++ -std=c++11 -fPIC -shared -o ./utils/combine_flow.so ./utils/combine_flow.cpp')
cbf = CDLL(path)
cbf.combine_flow.argtypes = [POINTER(c_double), c_int, POINTER(c_int), POINTER(c_double)]
cbf.combine_flow.restype = None

ph_images = tf.placeholder(tf.float32, shape = (None, 2, None, None, 3))
net = PWCDCNet()
tf_predicted_flow, _ = net(ph_images[:,0], ph_images[:,1])

saver_flow = tf.train.Saver()

ph_input_images = tf.placeholder(tf.float32, [1,res,res,3])
ph_input_flows = tf.placeholder(tf.float32, [1,res,res,2])
ph_input_feature = tf.placeholder(tf.float32, [1, res, res, 64])
ph_input_feature_secondary = tf.placeholder(tf.float32, [1, 64])

model = Model(ph_input_images, ph_input_flows, ph_input_feature, ph_input_feature_secondary, 1)
model.create_model()

saver_svbrdf = tf.train.Saver(var_list=[v for v in tf.global_variables() if 'trainableModel' in v.name])
op_init = tf.variables_initializer(var_list=tf.global_variables())

config = tf.ConfigProto()
config.allow_soft_placement=True
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(op_init)
saver_svbrdf.restore(sess, args.model_svbrdf)

def flow2full(flow, intrinsic):
    remapflow = np.zeros_like(flow.astype(np.float64))
    remapflow[:,:,0] = -flow[:,:,0] + np.arange(res0)
    remapflow[:,:,1] = -flow[:,:,1] + np.arange(res0)[:,np.newaxis]
    
    valid = np.where(np.abs(remapflow[...,0]) < 1e6)
    objectPoints = np.stack([valid[1]/res0*2-1, valid[0]/res0*2-1, [0.0]*len(valid[0])], 1)
    imagePoints = remapflow[valid]
    
    if len(objectPoints) >= 10:
        retval, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, intrinsic, None)

        objectPoints_1k = np.reshape(np.concatenate([np.transpose(np.mgrid[0:res, 0:res], (2,1,0)) / 512 - 1, np.zeros([res,res,1])], axis=-1), (res*res,3))
        imagePoints_1k, _ = cv2.projectPoints(objectPoints_1k, rvec, tvec, intrinsic * 4, None)
        remapflow1k = np.reshape(imagePoints_1k, (res,res,2))
        flow1k = np.zeros([res,res,2], np.float32)
        flow1k[:,:,0] = -remapflow1k[:,:,0] + np.arange(res)
        flow1k[:,:,1] = -remapflow1k[:,:,1] + np.arange(res)[:,np.newaxis]

        return flow1k

    else:
        logger.info('No enough points for solvePnP, returning upsampled')
        return upsample2(flow, 4) * 4


if os.path.isdir(args.input):
    files = glob.glob('{}/{}.npy') + glob.glob('{}/{}.avi')
elif args.input.endswith('.npy') or args.input.endswith('.avi'):
    files = [args.input]
else:
    print('Failed to load input.')
    files = []

for f in files:
    print('Processing {}'.format(f))
    if f.endswith('.npy'):
        video = np.load(f)
    elif f.endswith('.avi'):
        cap = cv2.VideoCapture(f)
        video = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video.append(frame)
        cap.release()
    video = np.array(video)

    fms = len(video)
    ns = fms // fmstep + 1
    refs = ns - 1
    predicted_flows = np.zeros((fms-1,res0,res0,2), np.float32)

    video256 = downsample(video, 4) / 255
    saver_flow.restore(sess, args.model_adjacent)

    for k in range(fms-1):
        images = np.zeros((bs,2,res0,res0,3), np.float32)
        for l in range(bs):
            images[l,1] = video256[k*bs+l]
            images[l,0] = video256[k*bs+l+1]
        predicted_flow = sess.run(tf_predicted_flow, feed_dict={ph_images:images})
        predicted_flows[k*bs:(k+1)*bs] = predicted_flow
    

    saver_flow.restore(sess, args.model_refinement)
    downsampled_flow = np.zeros((res0,res0,2), np.float32)

    input_frames = np.array([video[k*fmstep] for k in range(ns)])
    features_secondary = np.zeros((ns,64), np.float32)

    for k in range(ns):
        if k == 0:
            ft, fs = sess.run([model.generator_output, model.generator_secondary_output], feed_dict={ph_input_images:((input_frames[k:(k+1)]/255)**2.2*2-1), ph_input_flows:np.zeros((1,res,res,2), np.float32)})
            max_feature = ft[0]
            max_feature_secondary = fs[0]
        else:
            input_flows = np.concatenate([downsampled_flow[np.newaxis,...], predicted_flows[((k-1)*fmstep):(k*fmstep)]], axis=0)
            flatten_input = input_flows.flatten().astype(np.float64)
            flatten_output_mask = np.zeros(res0*res0, int)
            flatten_output_flow = np.zeros(res0*res0*2, np.float64)
            cbf.combine_flow(flatten_input.ctypes.data_as(POINTER(c_double)), fmstep+1, flatten_output_mask.ctypes.data_as(POINTER(c_int)), flatten_output_flow.ctypes.data_as(POINTER(c_double)))
            output_flow = np.reshape(flatten_output_flow, (res0,res0,2))

            full_flow = flow2full(output_flow, intrinsic)
            frame_ref = warp_flow_bv(video[k*fmstep]/255, full_flow)
            
            coords = max_valid_noinit(frame_ref)
            if coords is None:
                print('none coords happened, discarding frame')
                downsampled_flow = downsample(full_flow / 4, 4)
                continue
            else:
                up, down, left, right = coords

            subframe_main = video[0][up:down,left:right] / 255
            subframe_ref = frame_ref[up:down,left:right]

            images = np.stack([subframe_ref, subframe_main], axis=0)[np.newaxis,...]
            predicted_flow = sess.run(tf_predicted_flow, feed_dict={ph_images:images})[0]

            adjusting_flow = np.ones((res,res,2))*1e20
            adjusting_flow[up:down,left:right] = predicted_flow

            adjusted_flow = full_flow + adjusting_flow
            downsampled_flow = downsample(flow2full(downsample(adjusted_flow / 4, 4), intrinsic) / 4, 4)

            ft, fs = sess.run([model.generator_output, model.generator_secondary_output], feed_dict={ph_input_images:((input_frames[k:(k+1)]/255)**2.2*2-1), ph_input_flows:adjusted_flow[np.newaxis,...]})
            max_feature = np.maximum(max_feature, ft[0])
            max_feature_secondary = np.maximum(max_feature_secondary, fs[0])

    svbrdf_pred = sess.run(model.output, feed_dict={ph_input_feature:max_feature[np.newaxis,...], ph_input_feature_secondary:max_feature_secondary[np.newaxis,...]})
    saveSVBRDF('{}/{}.png'.format(args.output, f.split('/')[-1][:-4]), svbrdf_pred)
