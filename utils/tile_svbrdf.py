import cv2
import numpy as np
import glob

files = np.sort(glob.glob('/path-to-inria-data/train/*.png'))
for i, file in enumerate(files):
	img = cv2.imread(file)
	svbrdf = np.concatenate([img[:,0:512,::-1], img[:,512:1024,::-1], img[:,1024:1536,::-1], img[:,1536:2048,::-1]], axis=-1)
	svbrdf_vflip = np.array(svbrdf[::-1])
	svbrdf_vflip[...,1] = 255 - svbrdf_vflip[...,1]
	svbrdf_hflip = np.array(svbrdf[:,::-1])
	svbrdf_hflip[...,0] = 255 - svbrdf_hflip[...,0]
	svbrdf_vhflip = np.array(svbrdf_hflip[::-1])
	svbrdf_vhflip[...,1] = 255 - svbrdf_vhflip[...,1]
	svbrdf_tilable = np.concatenate([np.concatenate([svbrdf, svbrdf_hflip], axis=1), np.concatenate([svbrdf_vflip, svbrdf_vhflip], axis=1)], axis=0)
	np.save('/path-to-write-training-data/InriaSVBRDF_npy/train/{}.npy'.format(i), svbrdf_tilable)
	#svbrdf_4k = np.tile(svbrdf_tilable, [4,4,1])