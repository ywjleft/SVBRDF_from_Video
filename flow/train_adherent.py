import os, argparse, time
import tensorflow as tf
import numpy as np
from functools import partial
from ctypes import *

from flow.model_flow import PWCDCNet
from flow.losses_flow import EPE, multiscale_loss, multirobust_loss
from utils.utils import vis_flow_pyramid

from flow.video_generator_flow import Video_Generator

parser = argparse.ArgumentParser()

parser.add_argument('-e', '--num_epochs', type = int, default = 50,
                    help = '# of epochs [100]')
parser.add_argument('-b', '--batch_size', type = int, default = 4,
                    help = 'Batch size [4]')
parser.add_argument('-r', '--resume', type = str, default = "/home/D/v-wenye/pwcnet/model_100epochs_ft_Chairs/model_50.ckpt",
                    help = 'Learned parameter checkpoint file [None]')
parser.add_argument('-gpuid', default='0')
parser.add_argument('-experiment_name', dest='experiment_name', default='train_adherent_')

# the following parameters are unnecessary to change in common cases
parser.add_argument('--num_levels', type = int, default = 6,
                    help = '# of levels for feature extraction [6]')
parser.add_argument('--search_range', type = int, default = 4,
                    help = 'Search range for cost-volume calculation [4]')
parser.add_argument('--warp_type', default = 'bilinear', choices = ['bilinear', 'nearest'],
                    help = 'Warping protocol, [bilinear] or nearest')
parser.add_argument('--use-dc', dest = 'use_dc', action = 'store_true',
                    help = 'Enable dense connection in optical flow estimator, [diabled] as default')
parser.add_argument('--no-dc', dest = 'use_dc', action = 'store_false',
                    help = 'Disable dense connection in optical flow estimator, [disabled] as default')
parser.set_defaults(use_dc = False)
parser.add_argument('--output_level', type = int, default = 4,
                    help = 'Final output level for estimated flow [4]')
parser.add_argument('--loss', default = 'multiscale', choices = ['multiscale', 'robust'],
                    help = 'Loss function choice in [multiscale/robust]')
parser.add_argument('--lr', type = float, default = 1e-4,
                    help = 'Learning rate [1e-4]')
parser.add_argument('--lr_scheduling', dest = 'lr_scheduling', action = 'store_true',
                    help = 'Enable learning rate scheduling, [enabled] as default')
parser.add_argument('--no-lr_scheduling', dest = 'lr_scheduling', action = 'store_false',
                    help = 'Disable learning rate scheduling, [enabled] as default')
parser.set_defaults(lr_scheduling = False)
parser.add_argument('--weights', nargs = '+', type = float,
                    default = [0.32, 0.08, 0.02, 0.01, 0.005],
                    help = 'Weights for each pyramid loss')
parser.add_argument('--gamma', type = float, default = 0.0004,
                    help = 'Coefficient for weight decay [4e-4]')
parser.add_argument('--epsilon', type = float, default = 0.02,
                    help = 'Small constant for robust loss [0.02]')
parser.add_argument('--q', type = float, default = 0.4,
                    help = 'Tolerance constant for outliear flow [0.4]')


args = parser.parse_args()
for key, item in vars(args).items():
    print(f'{key} : {item}')

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

datadir = '/home/E/v-wenye/corr_data'
svbrdf_adobe_path = os.path.join(datadir, 'AdobeStockSVBRDF_npy/valid')
np.random.seed(123)
train_ids = np.random.choice(1195, 1000, replace=False)
test_ids = np.array([i for i in range(1195) if not i in train_ids])
svbrdf_inria_path = os.path.join(datadir, 'InriaSVBRDF_npy/train')

path = './cal_flow_v2.so'
cal_flow = CDLL(path)
cal_flow.calculate_flow.argtypes = [POINTER(c_float), c_double, POINTER(c_float)]
cal_flow.calculate_flow.restype = None

mixfactor = 50


class Trainer(object):
    def __init__(self, args):
        self.args = args
        if not self.args.resume is None:
            np.random.seed(int(time.time()))

        config = tf.ConfigProto()
        config.allow_soft_placement=True
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)
        self._build_graph()
        self.vg = Video_Generator(256)
        
        self.logdir = '/home/D/v-wenye/exp/corr_flow/' + self.args.experiment_name

    def _build_graph(self):
        # Input images and ground truth optical flow definition
        with tf.name_scope('Data'):
            self.images = tf.placeholder(tf.float32, shape = (self.args.batch_size, 2, 256, 256, 3),
                                         name = 'images')
            images_0, images_1 = tf.unstack(self.images, axis = 1)
            self.flows_gt = tf.placeholder(tf.float32, shape = (self.args.batch_size, 256, 256, 2),
                                           name = 'flows')

        # Model inference via PWCNet
        model = PWCDCNet(num_levels = self.args.num_levels,
                         search_range = self.args.search_range,
                         warp_type = self.args.warp_type,
                         use_dc = self.args.use_dc,
                         output_level = self.args.output_level,
                         name = 'pwcdcnet')
        flows_final, self.flows = model(images_0, images_1)

        # Loss calculation
        with tf.name_scope('Loss'):
            if self.args.loss is 'multiscale':
                criterion = multiscale_loss
            else:
                criterion =\
                  partial(multirobust_loss, epsilon = self.args.epsilon, q = self.args.q)
            
            _loss = criterion(self.flows_gt, self.flows, self.args.weights)
            weights_l2 = tf.reduce_sum([tf.nn.l2_loss(var) for var in model.vars])
            self.loss = _loss + self.args.gamma*weights_l2

            epe = EPE(self.flows_gt, flows_final)

        # Gradient descent optimization
        with tf.name_scope('Optimize'):
            self.global_step = tf.train.get_or_create_global_step()
            if self.args.lr_scheduling:
                boundaries = [200000, 250000, 300000, 350000, 400000]
                values = [self.args.lr/(2**i) for i in range(len(boundaries)+1)]
                lr = tf.train.piecewise_constant(self.global_step, boundaries, values)
            else:
                lr = self.args.lr

            self.optimizer = tf.train.AdamOptimizer(learning_rate = lr)\
                             .minimize(self.loss, var_list = model.vars)
            with tf.control_dependencies([self.optimizer]):
                self.optimizer = tf.assign_add(self.global_step, 1)

        # Initialization
        self.saver = tf.train.Saver(model.vars)
        self.sess.run(tf.global_variables_initializer())
        if self.args.resume is not None:
            print(f'Loading learned model from checkpoint {self.args.resume}')
            self.saver.restore(self.sess, self.args.resume)

        # Summarize
        # Original PWCNet loss
        sum_loss = tf.summary.scalar('loss/pwc', self.loss)
        # EPE for both domains
        sum_epe = tf.summary.scalar('EPE/source', epe)
        # Merge summaries
        self.merged = tf.summary.merge([sum_loss, sum_epe])

        self.twriter = tf.summary.FileWriter(self.logdir+'/train', graph = self.sess.graph)
        self.vwriter = tf.summary.FileWriter(self.logdir+'/val', graph = self.sess.graph)

        print(f'Graph building completed, histories are logged in {self.logdir}')

            
    def train(self):

        if not os.path.exists(f'./{self.logdir}/model'):
            os.mkdir(f'./{self.logdir}/model')

        video_perm = np.array([], int)
        for i in range(self.args.num_epochs):
            dlist = np.concatenate([train_ids, np.random.choice(1590, 1000) + 10000])
            np.random.shuffle(dlist)
            video_perm = np.concatenate([video_perm, dlist])

        for e in range(self.args.num_epochs):
            # Training
            for i in range(1000//mixfactor):
                print('starting {}_{}'.format(e,i))
                dataset_input_frames = np.zeros((140*mixfactor,2,256,256,3), np.uint8)
                dataset_input_flows = np.zeros((140*mixfactor,256,256,2), np.float32)
                for k in range(mixfactor):
                    thisid = video_perm[e*1000+i*mixfactor+k]
                    if thisid < 10000:
                        file = '{}/{}.npy'.format(svbrdf_adobe_path, thisid)
                    else:
                        file = '{}/{}.npy'.format(svbrdf_inria_path, thisid - 10000)
                    video, gtsvbrdf, camera_params, fov = self.vg.generate_video(file, 141, 20, self.sess)

                    for l in range(140):
                        dataset_input_frames[k*140+l, 0] = video[l+1]
                        dataset_input_frames[k*140+l, 1] = video[l]

                    flatten_input = camera_params.flatten().astype(np.float32)
                    flatten_output = np.zeros((140*256*256*2), np.float32)
                    cal_flow.calculate_flow(flatten_input.ctypes.data_as(POINTER(c_float)), fov, flatten_output.ctypes.data_as(POINTER(c_float)))
                    dataset_input_flows[k*140:(k+1)*140] = np.reshape(flatten_output, (140,256,256,2))
                print('{},loaded'.format(time.asctime(time.localtime(time.time()))))

                frame_perm = np.random.permutation(140*mixfactor)
                lss = 0.0
                for k in range(140*mixfactor//self.args.batch_size):
                    images = dataset_input_frames[frame_perm[k*self.args.batch_size:(k+1)*self.args.batch_size]]/255
                    flows_gt = dataset_input_flows[frame_perm[k*self.args.batch_size:(k+1)*self.args.batch_size]]

                    _, g_step, ls = self.sess.run([self.optimizer, self.global_step, self.loss],
                                              feed_dict = {self.images: images,
                                                           self.flows_gt: flows_gt})

                    if np.isnan(ls):
                        if e == 0 and i == 0:
                            prev_model = self.args.resume
                        elif i == 0:
                            prev_model = f'./{self.logdir}/model/model_{e-1}_{1000//mixfactor-1}.ckpt'
                        else:
                            prev_model = f'./{self.logdir}/model/model_{e}_{i-1}.ckpt'
                        self.saver.restore(self.sess, prev_model)
                        print('{},resumed from nan'.format(time.asctime(time.localtime(time.time()))))

                        flow_set = []
                        flows_val = self.sess.run(self.flows, feed_dict = {self.images: images,
                                                                           self.flows_gt: flows_gt})
                        for l, flow in enumerate(flows_val):
                            upscale = 20/2**(self.args.num_levels-l)
                            flow_set.append(flow[0]*upscale)
                        flow_gt = flows_gt[0]
                        images_v = images[0]
                        vis_flow_pyramid(flow_set, flow_gt, images_v,
                                         f'./{self.logdir}/figure/nan_flow_{e}_{i}_{k}.pdf')
                    else:
                        lss += ls

                    if g_step%1000 == 0:
                        summary = self.sess.run(self.merged,
                                                feed_dict = {self.images: images,
                                                             self.flows_gt: flows_gt})
                        self.twriter.add_summary(summary, g_step)

                        # visualize estimated optical flow
                        if not os.path.exists(f'./{self.logdir}/figure'):
                            os.mkdir(f'./{self.logdir}/figure')
                        # Estimated flow values are downscaled, rescale them compatible to the ground truth
                        flow_set = []
                        flows_val = self.sess.run(self.flows, feed_dict = {self.images: images,
                                                                           self.flows_gt: flows_gt})
                        for l, flow in enumerate(flows_val):
                            upscale = 20/2**(self.args.num_levels-l)
                            flow_set.append(flow[0]*upscale)
                        flow_gt = flows_gt[0]
                        images_v = images[0]
                        vis_flow_pyramid(flow_set, flow_gt, images_v,
                                         f'./{self.logdir}/figure/flow_{e}_{i}_{k}.pdf')

                print('{},{}/{}: {}'.format(time.asctime(time.localtime(time.time())), e*1000//mixfactor+i, self.args.num_epochs*1000/mixfactor, lss/(140*mixfactor//self.args.batch_size)))

                self.saver.save(self.sess, f'./{self.logdir}/model/model_{e}_{i}.ckpt')
            
        self.twriter.close()
        self.vwriter.close()
        

if __name__ == '__main__':
    trainer = Trainer(args)
    trainer.train()