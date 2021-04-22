import numpy as np
import tensorflow as tf
import os
import math
import cv2
from PIL import Image, ImageDraw, ImageFont
import skimage.measure


def warp_feature_guide(img, flow, h, w, y0, x0, borderValue=-3.402823e38):
    remapflow = np.zeros_like(flow)
    remapflow[:,:,0] = -flow[:,:,0] + np.arange(w) - x0
    remapflow[:,:,1] = -flow[:,:,1] + (np.arange(h) - y0)[:,np.newaxis]
    result = cv2.remap(img, remapflow, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=[borderValue,]*4)
    return result

def warp_flow_bv(img, flow, borderValue=-3.402823e38):
    h, w = flow.shape[:2]
    remapflow = np.zeros_like(flow.astype(np.float32))
    remapflow[:,:,0] = -flow[:,:,0] + np.arange(w)
    remapflow[:,:,1] = -flow[:,:,1] + np.arange(h)[:,np.newaxis]
    result = cv2.remap(img, remapflow, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=[borderValue,]*3)
    return result

def warp_flow_guide(img, flow, h, w, y0, x0, borderValue=-3.402823e38):
    remapflow = np.zeros_like(flow.astype(np.float32))
    remapflow[:,:,0] = -flow[:,:,0] + np.arange(w) - x0
    remapflow[:,:,1] = -flow[:,:,1] + (np.arange(h) - y0)[:,np.newaxis]
    result = cv2.remap(img, remapflow, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=[borderValue,]*3)
    return result
    
def generate_normalized_random_direction(lowEps = 0.001, highEps = 0.05):
    r1 = np.random.rand() * (1 - highEps - lowEps) + lowEps
    r2 = np.random.rand()
    r = np.sqrt(r1)
    phi = 2 * np.pi * r2
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = np.sqrt(1.0 - np.square(r))
    finalVec = np.array([x, y, z])
    return finalVec

def saveBigImage(file, images, row=None, column=None, labels=None):
    images = np.array(images, np.uint8)
    if len(images) > 0:
        if column is None and row is None:
            column = 10
        if column is None:
            column = int(np.ceil(len(images) / row))
        if row is None:
            row = int(np.ceil(len(images) / column))
        height = np.shape(images[0])[0]
        width = np.shape(images[0])[1]
        bigimage = np.zeros((height*row,width*column,np.shape(images)[-1]), np.uint8)
        for i in range(len(images)):
            rowi = i // column
            columni = i % column
            bigimage[height*rowi:height*(rowi+1),width*columni:width*(columni+1),:] = images[i]
        if labels is None:
            cv2.imwrite(file, bigimage)
        else:
            img = Image.fromarray(bigimage, 'RGB')
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype('FreeMono.ttf',12)
            for i in range(len(images)):
                rowi = i // column
                columni = i % column
                text = '{}'.format(labels[i])
                if np.mean(images[i]) < 128:
                    draw.multiline_text((columni*width+10,rowi*height+10), text, 'white', font)
                else:
                    draw.multiline_text((columni*width+10,rowi*height+10), text, 'black', font)
            img.save(file, 'JPEG')

def saveResultImage_svbrdf(prefix, input_frames, input_flows, svbrdf_pred, svbrdf_gt):
    
    if len(np.shape(input_frames)) == 5:
        input_frames = input_frames[0]
    if len(np.shape(input_flows)) == 5:
        input_flows = input_flows[0]
    if len(np.shape(svbrdf_pred)) == 5:
        svbrdf_pred = svbrdf_pred[0]
    if len(np.shape(svbrdf_gt)) == 5:
        svbrdf_gt = svbrdf_gt[0]
    res = np.shape(input_frames)[1]
    images = np.zeros((12,res,res,3), np.uint8)

    images[0] = input_frames[0]
    images[1] = input_frames[1]
    images[2] = input_frames[-1]
    images[3] = warp_flow(input_frames[1], input_flows[1])

    images[4] = (svbrdf_pred[...,5:2:-1]+1)/2 * 255
    images[5] = (svbrdf_pred[...,2::-1]+1)/2 * 255
    images[6] = (svbrdf_pred[...,11:8:-1]+1)/2 * 255
    images[7] = (svbrdf_pred[...,8:5:-1]+1)/2 * 255

    images[8] = (svbrdf_gt[...,5:2:-1]+1)/2 * 255
    images[9] = (svbrdf_gt[...,2::-1]+1)/2 * 255
    images[10] = (svbrdf_gt[...,11:8:-1]+1)/2 * 255
    images[11] = (svbrdf_gt[...,8:5:-1]+1)/2 * 255

    saveBigImage('{}.jpg'.format(prefix), images, 3, 4)


def saveResultRender_synthetic_novelview(prefix, input_frames, flow_pred, svbrdf_pred, svbrdf_gt, rendered, cmp_rerenders, cmp_refs, gt_rerenders):
    res = np.shape(input_frames)[1]
    cnt = len(cmp_rerenders)
    images = np.zeros(((cnt+4)*3,res,res,3), np.uint8)

    input_frames = input_frames[...,::-1]
    images[0] = input_frames[0]
    if len(input_frames) > 1:
        for i in range(1, min(cnt+4, len(input_frames))):
            images[i] = warp_flow(input_frames[i], flow_pred[i])
        '''
        images[1] = input_frames[1]
        images[2] = input_frames[-1]
        images[3] = warp_flow(input_frames[-1], flow_pred[-1])
        '''
    
    #images[4] = input_frames[0]
    #for i in range(cnt-1):
    #    images[5+i] = cmp_refs[i][...,::-1]

    images[cnt+4] = (svbrdf_pred[0,:,:,5:2:-1]+1)/2 * 255
    images[cnt+5] = (svbrdf_pred[0,:,:,2::-1]+1)/2 * 255
    images[cnt+6] = (svbrdf_pred[0,:,:,11:8:-1]+1)/2 * 255
    images[cnt+7] = (svbrdf_pred[0,:,:,8:5:-1]+1)/2 * 255

    for i in range(cnt):
        images[cnt+8+i] = cmp_rerenders[i,...,::-1]
    
    images[2*cnt+8] = (svbrdf_gt[0,:,:,5:2:-1]+1)/2 * 255
    images[2*cnt+9] = (svbrdf_gt[0,:,:,2::-1]+1)/2 * 255
    images[2*cnt+10] = (svbrdf_gt[0,:,:,11:8:-1]+1)/2 * 255
    images[2*cnt+11] = (svbrdf_gt[0,:,:,8:5:-1]+1)/2 * 255

    for i in range(cnt):
        images[2*cnt+12+i] = gt_rerenders[i,...,::-1]

    saveBigImage('{}.jpg'.format(prefix), images, 3, cnt+4)


def saveSVBRDF(prefix, svbrdfs):

    cnt,h,w = np.shape(svbrdfs)[0:3]
    images = np.zeros((cnt*4,h,w,3), np.uint8)

    for i in range(cnt):
        images[4*i+0] = (svbrdfs[i,:,:,5:2:-1]+1)/2 * 255
        images[4*i+1] = (svbrdfs[i,:,:,2::-1]+1)/2 * 255
        images[4*i+2] = (svbrdfs[i,:,:,11:8:-1]+1)/2 * 255
        images[4*i+3] = (svbrdfs[i,:,:,8:5:-1]+1)/2 * 255

    saveBigImage('{}.jpg'.format(prefix), images, cnt, 4)


# gradually expand from center to get valid area
def max_valid(img, initial=256, step=8, return_coords=False):
    orih, oriw = np.shape(img)[:2]
    right = oriw // 2 + initial // 2
    up = orih // 2 - initial // 2
    left = oriw // 2 - initial // 2
    down = orih // 2 + initial // 2
    expand_right = True
    expand_up = True
    expand_left = True
    expand_down = True
    while expand_right or expand_up or expand_left or expand_down:
        if expand_right:
            if right + step <= oriw and np.all(img[up:down, right:(right+step)] > -1e20):
                right += step
            else:
                expand_right = False
        if expand_up:
            if up - step >= 0 and np.all(img[(up-step):up, left:right] > -1e20):
                up -= step
            else:
                expand_up = False
        if expand_left:
            if left - step >= 0 and np.all(img[up:down, (left-step):left] > -1e20):
                left -= step
            else:
                expand_left = False
        if expand_down:
            if down + step <= orih and np.all(img[down:(down+step), left:right] > -1e20):
                down += step
            else:
                expand_down = False
    print('{},{},{},{}'.format(up,down,left,right))
    if return_coords:
        return img[up:down, left:right], [up,down,left,right]
    else:
        return img[up:down, left:right]


# gradually expand from center to get valid area
def max_valid_noinit(img, step=1):
    orih, oriw = np.shape(img)[:2]
    valid = img[...,0] >= 0
    #print('valid: {}'.format(np.sum(valid)))
    if np.sum(valid) < 100:
        return None
    ygrid, xgrid = np.mgrid[0:orih, 0:oriw]
    up = int(np.sum(valid*ygrid) / np.sum(valid))
    left = int(np.sum(valid*xgrid) / np.sum(valid))

    if not valid[up,left]:
        print('special')
        flag = False
        for j in range(orih):
            for i in range(oriw):
                if valid[j,i]:
                    up = j
                    left = i
                    flag = True
                    break
            if flag:
                break
    
    #print('initial: {},{}'.format(up,left))
    down = up + 1
    right = left + 1

    expand_right = True
    expand_up = True
    expand_left = True
    expand_down = True
    while expand_right or expand_up or expand_left or expand_down:
        if expand_right:
            if right + step <= oriw and np.all(img[up:down, right:(right+step)] > -1e20):
                right += step
            else:
                expand_right = False
        if expand_up:
            if up - step >= 0 and np.all(img[(up-step):up, left:right] > -1e20):
                up -= step
            else:
                expand_up = False
        if expand_left:
            if left - step >= 0 and np.all(img[up:down, (left-step):left] > -1e20):
                left -= step
            else:
                expand_left = False
        if expand_down:
            if down + step <= orih and np.all(img[down:(down+step), left:right] > -1e20):
                down += step
            else:
                expand_down = False
    #print('valid area: {},{},{},{}'.format(up,down,left,right))
    return up, down, left, right

def downsample(input_array, size=2):
	blocksize = np.ones(len(np.shape(input_array)), int)
	blocksize[-2] = size
	blocksize[-3] = size
	return skimage.measure.block_reduce(input_array, tuple(blocksize), np.mean)

def upsample2(input_array, size=2):
    h,w,_ = np.shape(input_array)
    return np.stack([cv2.resize(input_array[...,0], (w*size,h*size)), cv2.resize(input_array[...,1], (w*size,h*size))], axis=-1)

def upsample2_array(input_array, size=2):
    cnt,h,w,_ = np.shape(input_array)
    return np.stack([np.stack([cv2.resize(input_array[i,...,0], (w*size,h*size)), cv2.resize(input_array[i,...,1], (w*size,h*size))], axis=-1) for i in range(cnt)])

def resize2(input_array, size):
    return np.stack([cv2.resize(input_array[...,0], (size[1], size[0])), cv2.resize(input_array[...,1], (size[1], size[0]))], axis=-1)


    
import matplotlib
matplotlib.use('Agg')
from pylab import box
import matplotlib.pyplot as plt

def vis_flow_pyramid(flow_pyramid, flow_gt = None, images = None, filename = './flow.png'):
    num_contents = len(flow_pyramid) + int(flow_gt is not None) + int(images is not None)*2
    fig = plt.figure(figsize = (12, 15*num_contents))

    fig_id = 1

    if images is not None:
        plt.subplot(1, num_contents, fig_id)
        plt.imshow(images[0])
        plt.tick_params(labelbottom = False, bottom = False)
        plt.tick_params(labelleft = False, left = False)
        plt.xticks([])
        box(False)
        fig_id += 1

        plt.subplot(1, num_contents, num_contents)
        plt.imshow(images[1])
        plt.tick_params(labelbottom = False, bottom = False)
        plt.tick_params(labelleft = False, left = False)
        plt.xticks([])
        box(False)
            
    for flow in flow_pyramid:
        plt.subplot(1, num_contents, fig_id)
        plt.imshow(vis_flow(flow))
        plt.tick_params(labelbottom = False, bottom = False)
        plt.tick_params(labelleft = False, left = False)
        plt.xticks([])
        box(False)

        fig_id += 1

    if flow_gt is not None:
        plt.subplot(1, num_contents, fig_id)
        plt.imshow(vis_flow(flow_gt))
        plt.tick_params(labelbottom = False, bottom = False)
        plt.tick_params(labelleft = False, left = False)
        plt.xticks([])
        box(False)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches = 'tight', pad_inches = 0.1)
    plt.close()
