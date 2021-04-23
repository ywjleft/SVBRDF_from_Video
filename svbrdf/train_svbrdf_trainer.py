import tensorflow as tf
import numpy as np
import argparse, os, logging, time
from svbrdf.model_svbrdf import Model
from svbrdf.losses_svbrdf import Loss
from utils.utils import saveResultImage_svbrdf

parser = argparse.ArgumentParser()

parser.add_argument('-epochs', dest='epochs', type=int, default=50, help='number of epochs of training')
parser.add_argument('-loadModel', dest='loadModel', default=None, help='if you want to load a model and continue the training, input the path to the model here')
parser.add_argument('-startEpoch', dest='startEpoch',  type=int, default=0, help='the number of the first epoch, it only affects output')
parser.add_argument('-gpuid', dest='gpuid', default='0', help='the value for CUDA_VISIBLE_DEVICES')
parser.add_argument('-experiment_name', dest='experiment_name', default='train_svbrdf_')

args = parser.parse_args()

tempdir = '/ramdisk/'
# the trainer runs cooperatively with the loader, and exchange data through a ramdisk specified here
# Mount the path to physical memory greatly improves training speed, e.g., "mount -t tmpfs -o size=30G tmpfs /ramdisk" 

outputdir = '/parent-directory-to-output/' + args.experiment_name # the path to output results and logs

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

os.makedirs(os.path.join(outputdir, 'images_train'), exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(outputdir, 'training_log.txt'))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

logger.info(args)


res = 256
mixfactor = 50 # the number of video sequences that the loader sends to the trainer at one time, the extracted training pairs will be mixed together during training

ph_input_frames = tf.placeholder(tf.float32, [None,res,res,3])
ph_input_flows = tf.placeholder(tf.float32, [None,res,res,2])
ph_svbrdf_gt = tf.placeholder(tf.float32, [1,res,res,12]) # normal diffuse roughness specular ?

model = Model(ph_input_frames, ph_input_flows, 1, pooling_type='max')
model.create_model()
loss = Loss('mixed', model.output, ph_svbrdf_gt, res, 1, tf.placeholder(tf.float64, shape=(), name="lr"), True)
loss.createLossGraph()
loss.createTrainVariablesGraph()

saver = tf.train.Saver(max_to_keep=1)
op_init = tf.variables_initializer(var_list=tf.global_variables())

config = tf.ConfigProto()
config.allow_soft_placement=True
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(op_init)

if not args.loadModel is None:
    saver.restore(sess, args.loadModel)
    np.random.seed(int(time.time()))
    logger.info('resumed from: ' + args.loadModel)


for i in range(args.startEpoch, args.epochs):
    logger.info('Start epoch {}/{}'.format(i, args.epochs))
    for j in range(2000//mixfactor):

        while not os.path.isfile(tempdir + '{}_{}_ns.npy'.format(i,j)):
            time.sleep(1)

        dataset_input_frames = np.load(tempdir + '{}_{}_frames.npy'.format(i,j))
        dataset_input_flows = np.load(tempdir + '{}_{}_flows.npy'.format(i,j))
        dataset_svbrdf_gt = np.load(tempdir + '{}_{}_svbrdfs.npy'.format(i,j))
        dataset_ns = np.load(tempdir + '{}_{}_ns.npy'.format(i,j))
        os.remove(tempdir + '{}_{}_frames.npy'.format(i,j))
        os.remove(tempdir + '{}_{}_flows.npy'.format(i,j))
        os.remove(tempdir + '{}_{}_svbrdfs.npy'.format(i,j))
        os.remove(tempdir + '{}_{}_ns.npy'.format(i,j))
        print('loaded')

        frame_perm = np.random.permutation(mixfactor*20)
        total_loss = 0.0
        for k in range(mixfactor*20):
            ns = dataset_ns[frame_perm[k]]
            input_frames = dataset_input_frames[frame_perm[k]][:ns]
            input_flows = dataset_input_flows[frame_perm[k]][:ns]
            svbrdf_gt = dataset_svbrdf_gt[frame_perm[k]][np.newaxis,...]
            lr = min((i*20000+j*mixfactor*20+k)*0.0005*0.00002, 0.00002)
            feeddict = {ph_input_frames:np.reshape((input_frames/255)**2.2*2-1,[ns,res,res,3]), ph_input_flows:np.reshape(input_flows,[ns,res,res,2]), ph_svbrdf_gt:svbrdf_gt, loss.lr:lr}
            if k >= 10:
                _, ls = sess.run([loss.trainOp, loss.lossValue], feed_dict=feeddict)
                total_loss += np.mean(ls)
            else:
                _, ls, svbrdf_pred = sess.run([loss.trainOp, loss.lossValue, model.output], feed_dict=feeddict)
                saveResultImage_svbrdf(os.path.join(outputdir, 'images_train/{}_{}_{}'.format(i,j,k)), input_frames[...,::-1], input_flows, svbrdf_pred, svbrdf_gt)
                total_loss += np.mean(ls)
                
        logger.info('Iteration {}, loss: {}'.format(i*40000+j*mixfactor*20+k, total_loss/(mixfactor*20)))
    saver.save(sess, os.path.join(outputdir, '{}.ckpt'.format(i)))


