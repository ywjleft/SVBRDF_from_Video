# This test script reconstructs high-resolution SVBRDF maps. It uses a macro-view photo as guidance. 
# The position of the first video frame in the guidance photo needs to be specified manually. 

import tensorflow as tf
import numpy as np
import argparse, os, cv2
from ctypes import *

from flow.model_flow import PWCDCNet
from svbrdf.model_svbrdf_bigmap_nowarp import Model
from utils.utils import max_valid_noinit, max_valid, warp_flow_guide, warp_flow_bv, warp_feature_guide, downsample, resize2, upsample2_array, saveSVBRDF

parser = argparse.ArgumentParser()

'''
The input contains of 3 parts, a 1024x1024 video in .npy or .avi, a guidance photo in .png or .jpg, and the first frame position. 
The first frame position can be specified in the following two ways:
- A .npy file containing a 1D int array with length 4, specifying y0,y1,x0,x1
- A string with the format y0,y1,x0,x1
'''

parser.add_argument('-input_video', dest='input_video', default='/home/D/v-wenye/svbrdfvideo/npy1k_7/9.npy')
parser.add_argument('-input_guidance', dest='input_guidance', default='/home/D/v-wenye/svbrdfvideo/npy1k_7/9.png')
parser.add_argument('-input_position', dest='input_position', default='96,761,1910,2575', help='the position of the first video frame in the guidance photo')
parser.add_argument('-output', dest='output', default='/home/F/v-wenye/exp/test', help='the folder to output svbrdf maps')
parser.add_argument('-gpuid', dest='gpuid', default='0', help='the value for CUDA_VISIBLE_DEVICES')

parser.add_argument('-model_adjacent', dest='model_adjacent', default='/home/F/v-wenye/flow_model_final/adjacent_49_19.ckpt')
parser.add_argument('-model_refinement', dest='model_refinement', default='/home/F/v-wenye/flow_model_final/distant_49_19.ckpt')
parser.add_argument('-model_svbrdf', dest='model_svbrdf', default='/home/D/v-wenye/exp/corr_svbrdf/v7p_norottrans_further/51_9.ckpt')

args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

os.makedirs(args.output, exist_ok=True)

bs = 1
res = 256
res1 = 1024
fmstep = 5

fov = 30 / 180 * np.pi
f = 0.5 * res / np.tan(fov/2)
c = 0.5 * res
intrinsic = np.array([[f,0,c],[0,f,c],[0,0,1]])

path = './utils/combine_flow_bigmap.so'
if not os.path.isfile(path):
    os.system('g++ -std=c++11 -fPIC -shared -o ./utils/combine_flow_bigmap.so ./utils/combine_flow_bigmap.cpp')
cbf_bigmap = CDLL(path)
cbf_bigmap.combine_flow.argtypes = [POINTER(c_double), c_int, c_int, c_int, c_int, POINTER(c_double), c_int, POINTER(c_double)] # bigmap flow, h, w, flows, cnt, out flow
cbf_bigmap.combine_flow.restype = None


with tf.device('/device:GPU:0'):
    ph_images = tf.placeholder(tf.float32, shape = (None, 2, None, None, 3))
    net = PWCDCNet()
    tf_predicted_flow, _ = net(ph_images[:,0], ph_images[:,1])

saver_flow = tf.train.Saver()

with tf.device('/device:GPU:0'):
    ph_input_images = tf.placeholder(tf.float32, [1, res1, res1, 3])
with tf.device('/device:CPU:0'):
    ph_input_feature = tf.placeholder(tf.float32, [1, None, None, 64])
    ph_input_feature_secondary = tf.placeholder(tf.float32, [1, 64])

model = Model(ph_input_images, ph_input_feature, ph_input_feature_secondary, 1, useCPU=True)
model.create_model()

saver_svbrdf = tf.train.Saver(var_list=[v for v in tf.global_variables() if 'trainableModel' in v.name])

op_init = tf.variables_initializer(var_list=tf.global_variables())

config = tf.ConfigProto()
config.allow_soft_placement=True
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(op_init)

saver_svbrdf.restore(sess, args.model_svbrdf)

def flow2full(flow, y0, x0):
    h, w, _ = np.shape(flow)
    remapflow = np.zeros_like(flow.astype(np.float64))
    remapflow[:,:,0] = -flow[:,:,0] + np.arange(w) - x0
    remapflow[:,:,1] = -flow[:,:,1] + (np.arange(h) - y0)[:,np.newaxis]
    
    valid = np.where(np.abs(remapflow[...,0]) < 1e6)
    objectPoints = np.stack([(valid[1]-x0)/res1*2-1, (valid[0]-y0)/res1*2-1, [0.0]*len(valid[0])], 1)
    imagePoints = remapflow[valid]
    
    if len(objectPoints) < 10:
        print('No enough points in solvePnP, returning original flow!')
        return flow
    
    retval, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, intrinsic, None)

    objectPoints_1k = np.reshape(np.concatenate([np.transpose(np.mgrid[(-x0):(w-x0), (-y0):(h-y0)], (2,1,0)) / 512 - 1, np.zeros([h,w,1])], axis=-1), (h*w,3))
    imagePoints_1k, _ = cv2.projectPoints(objectPoints_1k, rvec, tvec, intrinsic, None)
    remapflow1k = np.reshape(imagePoints_1k, (h,w,2))
    flow1k = np.zeros([h,w,2], np.float32)
    flow1k[:,:,0] = -remapflow1k[:,:,0] + np.arange(w) - x0
    flow1k[:,:,1] = -remapflow1k[:,:,1] + (np.arange(h) - y0)[:,np.newaxis]
    return flow1k


if args.input_video.endswith('.npy'):
    video = np.load(args.input_video)
elif args.input_video.endswith('.avi'):
    cap = cv2.VideoCapture(args.input_video)
    video = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video.append(frame)
    cap.release()
video = np.array(video)

video256 = downsample(video, 4) / 255
fms = len(video)
ns = fms // fmstep
refs = ns - 1

guidance = cv2.imread(args.input_guidance)[...,::-1]

if args.input_position.endswith('.npy'):
    coords = np.load(args.input_position)
else:
    coords = np.array(list(map(int, args.input_position.split(','))))

y0,y1,x0,x1 = coords
resmul = res1 / ((y1+x1-y0-x0) / 2)
orih, oriw = np.shape(guidance)[:2]
h = int(orih * resmul)
w = int(oriw * resmul)
print('{},{},{}'.format(resmul,h,w))
guidance = cv2.resize(guidance, (w,h))
y0,y1,x0,x1 = (coords * resmul).astype(int)
y1 = y0 + res1
x1 = x0 + res1


predicted_flows = np.zeros((fms-1,res,res,2), np.float32)
saver_flow.restore(sess, args.model_adjacent)

for k in range(fms-1):
    images = np.zeros((bs,2,res,res,3), np.float32)
    for l in range(bs):
        images[l,1] = video256[k*bs+l]
        images[l,0] = video256[k*bs+l+1]
    predicted_flow = sess.run(tf_predicted_flow, feed_dict={ph_images:images})
    predicted_flows[k*bs:(k+1)*bs] = predicted_flow

saver_flow.restore(sess, args.model_refinement)

max_feature = np.full((h, w, 64), -3.402823e38, np.float32)
max_feature_secondary = np.full(64, -3.402823e38, np.float32)

for i in range(2):
    sum_guidance = np.zeros((h,w,3), float)
    sum_cnt = np.zeros((h,w,1), int)
    for k in range(ns):
        print('processing frame {}/{}'.format(k, ns))
        if k > 0:
            flatten_bigmap_input = rectified_flow.astype(np.float64).flatten()
            input_flows = upsample2_array(predicted_flows[((k-1)*fmstep):(k*fmstep)], 4) * 4
            flatten_input = input_flows.astype(np.float64).flatten()
            flatten_output_flow = np.zeros(h*w*2, np.float64)
            cbf_bigmap.combine_flow(flatten_bigmap_input.ctypes.data_as(POINTER(c_double)), h, w, y0, x0, flatten_input.ctypes.data_as(POINTER(c_double)), fmstep, flatten_output_flow.ctypes.data_as(POINTER(c_double)))
            output_flow = np.reshape(flatten_output_flow, (h,w,2))
            print(np.sum(np.abs(output_flow[...,0]) < 1e6))

            full_flow = flow2full(output_flow, y0, x0)

            frame_ref = warp_flow_guide(video[k*fmstep]/255, full_flow, h, w, y0, x0)

            coords = max_valid_noinit(frame_ref)
            if coords is None:
                adjusted_flow = 1e20
                print("continue1")
                continue
            else:
                up, down, left, right = coords

            subframe_main = guidance[up:down,left:right] / 255
            subframe_ref = frame_ref[up:down,left:right]

        else:
            up, down, left, right = [y0, y1, x0, x1]
            subframe_main = guidance[up:down,left:right] / 255
            subframe_ref = video[0] / 255

        # first on 256 resolution
        images = downsample(np.stack([subframe_ref, subframe_main], axis=0), 4)[np.newaxis,...]
        predicted_flow = sess.run(tf_predicted_flow, feed_dict={ph_images:images})[0]
        upsampled_flow = resize2(predicted_flow * 4, (down-up, right-left))
        subframe_ref_warped = warp_flow_bv(subframe_ref, upsampled_flow)
        coords1 = max_valid_noinit(subframe_ref_warped)
        if coords1 is None:
            print("continue2")
            continue
        up1, down1, left1, right1 = coords1
        subframe_main1 = subframe_main[up1:down1,left1:right1]
        subframe_ref1 = subframe_ref_warped[up1:down1,left1:right1]
        images1 = np.stack([subframe_ref1, subframe_main1], axis=0)[np.newaxis,...]
        predicted_flow1 = sess.run(tf_predicted_flow, feed_dict={ph_images:images1})[0]
        adjusting_flow1s = np.ones_like(upsampled_flow)*1e20
        adjusting_flow1s[up1:down1,left1:right1] = predicted_flow1
        
        adjusting_flow1 = np.ones((h,w,2))*1e20
        adjusting_flow1[up:down,left:right] = adjusting_flow1s
        adjusting_flow = np.ones((h,w,2))*1e20
        adjusting_flow[up:down,left:right] = upsampled_flow
        
        if k > 0:
            adjusted_flow = adjusting_flow1 + adjusting_flow + full_flow
        else:
            adjusted_flow = adjusting_flow1 + adjusting_flow
        
        rectified_flow = flow2full(adjusted_flow, y0, x0)

        if i == 1:
            f, fs = sess.run([model.generator_output, model.generator_secondary_output], feed_dict={ph_input_images:((video[k*fmstep][np.newaxis,...]/255)**2.2*2-1)})
            warped_feature = warp_feature_guide(f[0], adjusted_flow.astype(np.float32), h, w, y0, x0)
            max_feature = np.maximum(max_feature, warped_feature)
            max_feature_secondary = np.maximum(max_feature_secondary, fs[0])

        warped_img = warp_flow_guide(video[k*fmstep], adjusted_flow.astype(np.float32), h, w, y0, x0, 0)
        cbmap = 1 - (np.mean(warped_img, axis=-1) == 0)[...,np.newaxis]
        sum_cnt += cbmap
        sum_guidance += warped_img
    
    guidance_new = np.zeros((h,w,3), float)
    for yy in range(h):
        for xx in range(w):
            if sum_cnt[yy,xx] == 0:
                guidance_new[yy,xx] = guidance[yy,xx]
            else:
                guidance_new[yy,xx] = sum_guidance[yy,xx] / sum_cnt[yy,xx]
    guidance = guidance_new.copy()
    cv2.imwrite('{}/{}_guidance_{}.png'.format(args.output, args.input_video.split('/')[-1][:-4], i), guidance[...,::-1])

max_feature_valid = max_valid(max_feature)

svbrdfs = sess.run(model.output, feed_dict={ph_input_feature:max_feature_valid[np.newaxis,...], ph_input_feature_secondary:max_feature_secondary[np.newaxis,...]})

saveSVBRDF('{}/{}.png'.format(args.output, args.input_video.split('/')[-1][:-4]), svbrdfs)
