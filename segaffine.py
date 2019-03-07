import os, os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import random as rn
from skimage.transform import warp, AffineTransform, SimilarityTransform
import flow_utils as fu

from PIL import Image


WARP_CONST = -12345

def warp_bg(bg, w, h):

    X, Y = np.meshgrid(np.arange(0, bg.shape[1]), np.arange(0, bg.shape[0]))
    grid = np.dstack((X, Y))

    while True:

        tx = param_gen(4, 0, 1.3, -40, 40, 1)
        ty = param_gen(4, 0, 1.3, -40, 40, 1)
        an = param_gen(2, 0, 1.3, -10, 10, 0.3) # in degree
        sx = param_gen(2, 1, 0.1, 0.93, 1.07, 0.6)
        sy = param_gen(2, 1, 0.1, 0.93, 1.07, 0.6)

        tform = make_tf(w, h, tx, ty, an, sx, sy)
        #form = AffineTransform(translation=(tx, ty), scale=(sx, sy), rotation=radians(an))
        grid_ = warp(grid.astype(np.float32), tform.inverse, order=1,
                mode='constant', cval=-WARP_CONST)
        bg2 = warp(bg.astype(np.float32), tform.inverse, order=1)
        bgfl = grid_ - grid

        assert bgfl.shape[:-1] == bg2.shape[:-1], 'Warped BG and Flow has different size '\
                'is {} vs. {}'.format(str(bgfl.shape[:-1]), str(bg2.shape[:-1]))

        # cropping from invalid regions
        # getting the available regions
        mask = np.logical_and(  grid_[...,0] > WARP_CONST/10,
                                grid_[...,1] > WARP_CONST/10,
                                np.logical_or(  grid_[...,0] != 0,
                                                grid_[...,0] != 0))
        csy = np.cumsum(mask, axis=0)
        csx = np.cumsum(mask, axis=1)
        idx = np.logical_and(csy > h+5, csx > w+5,
                             np.cumsum(csx, axis=0) > ((h+5)*(w+5)))
        ys, xs = np.where(idx==True)

        # get the pair of x, y that gives no contamination from warping
        for y, x in zip(ys, xs):
            tmp = grid_[y-h:y,x-w:x,:]
            if (tmp < WARP_CONST/10).sum() == 0 and \
                np.all(tmp==(0.,0.),axis=2).sum() == 0:
                sy = y-h
                sx = x-w
                fl = bgfl[y-h:y,x-w:x,:]
                break
        if fl.min() > -100 and fl.max() < 100:
            break

    return  bg[y-h:y,x-w:x,:], bg2[y-h:y,x-w:x,:].astype(np.uint8), fl


def prepare_bg(bg, target_size, static=True):
    '''
        target_size: 2-tuple of target [height, width] in this order
    '''
    imh, imw = target_size
    bgh, bgw = bg.shape[:2]
    bgim = Image.fromarray(bg)

    hmax = max(bgh, imh)
    wmax = max(bgw, imw)
    # scale the background to at least 2 times larger than the image
    r = 1.5 #rn.uniform(2, 2.5) * max(float(hmax)/bgh, float(wmax)/bgw)
    bgim = bgim.resize((int(bgw*r), int(bgh*r)), Image.ANTIALIAS)
    bg = np.array(bgim)

    if not static:
        bg, bg2, bgfl = warp_bg(bg, imw, imh)
    else:
        # random position to crop
        sy, sx = rn.randint(0, bg.shape[0] - imh), rn.randint(0, bg.shape[1] - imw)
        # cropping
        bg = bg[sy:(sy+imh), sx:(sx+imw), :]
        bg2 = bg.copy()
        bgfl = np.zeros((bg.shape[0], bg.shape[1], 2))

    return bg, bg2, bgfl


def param_gen(k, mu, sigma, a, b, p):

    gamma = rn.gauss(mu, sigma)
    beta = np.random.binomial(1, p)
    return beta * max(min(np.sign(gamma) * abs(gamma)**k, b), a) + (1-beta)*mu


def make_tf(w, h, tx, ty, an, sx, sy):
    shift_y, shift_x = rn.randint(0, h), rn.randint(0, w)
    tf_shift = SimilarityTransform(translation=[-shift_x, -shift_y])
    tf_shift_inv = SimilarityTransform(translation=[shift_x, shift_y])
    tf_rotate = tf_shift + (SimilarityTransform(rotation=np.deg2rad(an)) + tf_shift_inv)
    tf_trans  = SimilarityTransform(translation=[tx, ty])
    tf_scale  = AffineTransform(scale=(sx,sy))

    return tf_trans + tf_rotate + tf_scale

def prepare_segment(im, mk):

    idx = mk != 0
    rmin, rmax = np.where(np.any(idx, axis=1))[0][[0, -1]]
    cmin, cmax = np.where(np.any(idx, axis=0))[0][[0, -1]]

    im1 = im[rmin:rmax, cmin:cmax, :]
    mk1 = mk[rmin:rmax, cmin:cmax]
    sh, sw = rmax - rmin, cmax - cmin # segment size

    ts = max(50, min(640, rn.gauss(200, 200))) # target size
    r  = float(ts) / max(sh, sw)
    print 'Random size: ', ts, ' Ratio: ', r
    im1 = np.array(Image.fromarray(im1).resize((int(sw*r), int(sh*r)),
                        Image.ANTIALIAS))
    mk1 = np.array(Image.fromarray(mk1).resize((int(sw*r), int(sh*r)),
                        Image.NEAREST))

    return im1, mk1

def run_1affine(im_path, mk_path, tw, th):

    im, mk = prepare_segment(np.array(Image.open(im_path)), np.array(Image.open(mk_path)))
    sh, sw, _ = im.shape

    # paste the segment to a larger frame, at random position x, y
    y, x = rn.randint(0, th-sh), rn.randint(0, tw-sw)
    im1 = np.zeros((th, tw, 3))
    mk1 = np.zeros((th, tw))
    im1[y:y+sh,x:x+sw,:] = im[:]
    mk1[y:y+sh,x:x+sw] = mk[:]

    im, mk = prepare_segment(im1, mk1)
    sh, sw, _ = im.shape

    im2 = np.zeros_like(im1)
    mk2 = np.zeros_like(mk1)
    flo = np.zeros(mk1.shape + (2,))

    pad = [ [int(th/2), int(th/2)], [int(tw/2), int(tw/2)]]
    im1 = np.pad(im1, pad+[[0, 0]], mode='constant', constant_values=0)
    mk1 = np.pad(mk1, pad, mode='constant', constant_values=0)
    gr  = np.dstack(np.meshgrid(np.arange(0, im1.shape[1]), np.arange(0, im1.shape[0])))
    gr[..., 0] -= int(tw/2)
    gr[..., 1] -= int(th/2)

    for s in np.unique(mk1):
        if s == 0:
            continue

        tx = param_gen(3, 0, 2.3, -120, 120, 1)
        ty = param_gen(3, 0, 2.3, -120, 120, 1)
        an = param_gen(2, 0, 2.3, -30, 30, 0.7) # in degree
        sx = param_gen(2, 1, 0.18, 0.8, 1.2, 0.7)
        sy = param_gen(2, 1, 0.18, 0.8, 1.2, 0.7)

        print tx, ty, an, sx, sy

        tform = make_tf(im1.shape[1], im1.shape[0], tx, ty, an, sx, sy)
        im_ = warp(im1.astype(np.float32), tform.inverse, order=1)[int(th/2):int(th/2)+th,int(tw/2):int(tw/2)+tw, :]
        mk_ = warp(mk1.astype(np.float32), tform.inverse, order=0)[int(th/2):int(th/2)+th,int(tw/2):int(tw/2)+tw]
        gr_ = warp(gr.astype(np.float32), tform.inverse, order=1,
                mode='constant', cval=-WARP_CONST)[int(th/2):int(th/2)+th,int(tw/2):int(tw/2)+tw, :]
        fl_ = gr_ - gr[int(th/2):int(th/2)+th,int(tw/2):int(tw/2)+tw, :]
        mk11 = mk1[int(th/2):int(th/2)+th,int(tw/2):int(tw/2)+tw]

        idx = mk_==s
        im2[idx] = im_[idx]
        mk2[idx] = mk_[idx]
        flo[mk11==s] = fl_[mk11==s]

        import pdb
        pdb.set_trace()
        plt.figure(), plt.imshow(fu.convert(flo)), plt.show(block=False)

    return  im1.astype(np.uint8), mk1.astype(np.uint8),\
            im2.astype(np.uint8), mk2.astype(np.uint8), flo


rn.seed(1020)

#tmp_paths = []
#bg_paths = []
#print "Scanning background directory... ",
#for bgroot, _, files in os.walk('data/naturedata'):
#    for f in files:
#        if '.PNG' not in f.upper() and \
#            '.JPG' not in f.upper() and  '.JPEG' not in f.upper():
#            continue
#        bg_paths.append(osp.join(bgroot, f))
#print "\t[Done]"
#
## load background
#while True:
#    if len(tmp_paths) == 0:
#        tmp_paths = sorted(bg_paths[:]) # copy
#    bgpath = rn.choice(tmp_paths)
#    tmp_paths.remove(bgpath)
#    try:
#        bgim = np.array(Image.open(bgpath))
#        if bgim.shape[2] == 3:
#            break
#    except:
#        pass
#    # if something wrong happens
#    bg_paths.remove(bgpath)

bgim = np.array(Image.open('data/naturedata/city__326.jpg'))


im_root = 'data/DAVIS/orgRGB'
mk_root = 'data/DAVIS/orgMasks'

im1, mk1, im2, mk2, fl = run_1affine(    osp.join(im_root, 'schoolgirls', '00001.jpg'),
                osp.join(mk_root, 'schoolgirls', '00001.png'),
                1024, 768)

# fit background to the image size
bgim1, bgim2, bgflo = prepare_bg(bgim, (768, 1024))

bgim1[mk1 != 0] = im1[mk1 != 0]
bgim2[mk2 != 0] = im2[mk2 != 0]

plt.figure()
plt.imshow(bgim1)
plt.figure()
plt.imshow(bgim2)
plt.figure()
plt.imshow(fu.convert(fl))
plt.show()


