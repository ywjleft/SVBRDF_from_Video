# weaker augmentation

import tensorflow as tf
import numpy as np
import argparse, os, time
from ctypes import *
from svbrdf.video_generator_svbrdf import Video_Generator

parser = argparse.ArgumentParser()

parser.add_argument('-epochs', dest='epochs', type=int, default=50, help='number of epochs of training')
parser.add_argument('-loadModel', dest='loadModel', default=None, help='if you want to load a model and continue the training, input the path to the model here')
parser.add_argument('-startEpoch', dest='startEpoch',  type=int, default=0, help='the number of the first epoch, it only affects output')
parser.add_argument('-gpuid', dest='gpuid', default='0', help='the value for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

tempdir = '/ramdisk/'
datadir = '/path-to-training-data' # the path containing training data

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

path = './utils/cal_flow_svbrdf_v4.so'
if not os.path.isfile(path):
    os.system('g++ -std=c++11 -fPIC -shared -o ./utils/cal_flow_svbrdf_v4.so ./utils/cal_flow_svbrdf_v4.cpp')
cal_flow_svbrdf = CDLL(path)
cal_flow_svbrdf.calculate_flow.argtypes = [POINTER(c_float), POINTER(c_int), c_double, POINTER(c_float)]
cal_flow_svbrdf.calculate_flow.restype = None

svbrdf_adobe_path = os.path.join(datadir, 'AdobeStockSVBRDF_npy/valid')
np.random.seed(123) # Since Adobe data originally does not have train/test separation, we use this seed to separate. 
train_ids = np.random.choice(1195, 1000, replace=False)
test_ids = np.array([i for i in range(1195) if not i in train_ids])
svbrdf_inria_path = os.path.join(datadir, 'InriaSVBRDF_npy/train')

res = 256
max_ns = 17 # number of input frames
min_ns = 5
mixfactor = 50

perlin_dataset_file = os.path.join(datadir, 'PerlinNoise/Perlin4k.npy')
vg = Video_Generator(res, perlin_dataset_file)

config = tf.ConfigProto()
config.allow_soft_placement=True
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

video_perm = np.array([], int)
for i in range(args.epochs):
    dlist = np.concatenate([train_ids, np.random.choice(1590, 1000) + 10000])
    np.random.shuffle(dlist)
    video_perm = np.concatenate([video_perm, dlist])

for i in range(args.startEpoch, args.epochs):
    for j in range(2000//mixfactor):
        
        if j == 0:
            if i > 0:
                while os.path.isfile(tempdir + '{}_{}_frames.npy'.format(i-1,2000//mixfactor-1)):
                    time.sleep(1)
        else:
            while os.path.isfile(tempdir + '{}_{}_frames.npy'.format(i,j-1)):
                time.sleep(1)
        print('start generation {}_{}'.format(i,j))

        dataset_input_frames = np.zeros((mixfactor*20,max_ns,res,res,3), np.uint8)
        dataset_input_flows = np.zeros((mixfactor*20,max_ns,res,res,2), np.float32)
        dataset_svbrdf_gt = np.zeros((mixfactor*20,res,res,12), np.float32)
        dataset_ns = np.zeros((mixfactor*20), int)

        for k in range(mixfactor):
            thisid = video_perm[i*2000+j*mixfactor+k]
            if thisid < 10000:
                file = '{}/{}.npy'.format(svbrdf_adobe_path, thisid)
            else:
                file = '{}/{}.npy'.format(svbrdf_inria_path, thisid - 10000)
            video, gtsvbrdf, camera_params, fov = vg.generate_video(file, 141, 20, sess)
            dataset_svbrdf_gt[k*20:(k+1)*20] = gtsvbrdf
            fmsteps = np.zeros(20, np.int32)
            for l in range(20):
                ns = np.random.randint(max_ns - min_ns + 1) + min_ns
                max_fmstep = 120 // (ns - 1)
                min_fmstep = max_fmstep // 2
                fmstep = np.random.randint(max_fmstep - min_fmstep + 1) + min_fmstep
                fmsteps[l] = fmstep
                dataset_ns[k*20+l] = ns
                dataset_input_frames[k*20+l,:ns] = video[l:(fmstep*ns+l):fmstep]
            flatten_input = camera_params.flatten().astype(np.float32)
            flatten_output = np.zeros((20*(max_ns-1)*res*res*2), np.float32)
            cal_flow_svbrdf.calculate_flow(flatten_input.ctypes.data_as(POINTER(c_float)), fmsteps.ctypes.data_as(POINTER(c_int)), fov, flatten_output.ctypes.data_as(POINTER(c_float)))
            dataset_input_flows[k*20:(k+1)*20] = np.concatenate([np.zeros((20,1,res,res,2), np.float32), np.reshape(flatten_output, (20,max_ns-1,res,res,2))], axis=1)

        np.save(tempdir + '{}_{}_frames.npy'.format(i,j), dataset_input_frames)
        np.save(tempdir + '{}_{}_flows.npy'.format(i,j), dataset_input_flows)
        np.save(tempdir + '{}_{}_svbrdfs.npy'.format(i,j), dataset_svbrdf_gt)
        np.save(tempdir + '{}_{}_ns.npy'.format(i,j), dataset_ns)
        print('finish generation {}_{}'.format(i,j))