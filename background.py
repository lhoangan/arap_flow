import sys
from math import pi, radians
import numpy as np, random as rn
from skimage.transform import warp, AffineTransform, SimilarityTransform
import matplotlib.pyplot as plt
from PIL import Image
import time

def convert(flow):
    UNKNOWN_FLOW_THRESH = 1e9

    h, w, nBands = flow.shape

    assert nBands == 2, 'flow_to_color: image must have two bands'

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999
    maxv = -999

    minu = 999
    minv = 999

    maxrad = -1

    # fix unknown flow
    idxUnknown = (np.abs(u) > UNKNOWN_FLOW_THRESH) | (np.abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknown] = 0
    v[idxUnknown] = 0

    maxu = max(maxu, np.max(u))
    maxv = max(maxv, np.max(v))
    minu = min(minu, np.min(u))
    minv = min(minv, np.min(v))

    rad = np.sqrt(np.square(u) + np.square(v))
    maxrad = max(maxrad, np.max(rad))

    #print ('max flow: %f

    u = u/(maxrad + np.finfo(np.float).eps)
    v = v/(maxrad + np.finfo(np.float).eps)

    # compute color
    img = compute_color(u, v)

    # unknown flow
    IDX = np.tile(idxUnknown[..., None], [1, 1, 3])
    img[IDX] = 0

    return img



def compute_color(u, v):

    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    img = np.zeros(list(u.shape)+ [3], dtype=np.uint8)

    color_wheel = make_color_wheel()
    ncols = color_wheel.shape[0]

    rad = np.sqrt(np.square(u) + np.square(v))

    a = np.arctan2(-v, -u) / pi

    fk = (a + 1) / 2 * (ncols - 1) # (-1):1 mapped to 0:ncols

    k0 = np.floor(fk).astype(np.int32)

    k1 = k0 + 1
    k1[k1 == ncols] = 1

    f = fk - k0

    for i in range(color_wheel.shape[1]):
        tmp = color_wheel[:, i]
        col0 = tmp[k0] / 255
        col1 = tmp[k1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1  # boolean type
        col[idx] = 1 - rad[idx] * (1 - col[idx])    # increase saturation with radius
        col[~idx] = col[~idx] * .75           # out of range

        img[:, :, i] = np.floor(255 * col * (1 - nanIdx)).astype(np.uint8)

    return img

# color enconding scheme
# followed the Sintel implementation of
# the color circle idea described at
# http://members.shaw.ca/quadibloc/other/colint.htm
def make_color_wheel():
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    color_wheel = np.zeros((ncols, 3)); # r, g, b

    col = 0
    # RY
    color_wheel[np.arange(RY), 0] = 255;
    color_wheel[np.arange(RY), 1] = np.floor(255 * np.arange(RY) / RY)
    col += RY

    # YG
    color_wheel[col + np.arange(YG), 0] = 255 - np.floor(255 * np.arange(YG) / YG)
    color_wheel[col + np.arange(YG), 1] = 255
    col += YG

    # GC
    color_wheel[col + np.arange(GC), 1] = 255
    color_wheel[col + np.arange(GC), 2] = np.floor(255 * np.arange(GC) / GC)
    col += GC

    # CB
    color_wheel[col + np.arange(CB), 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    color_wheel[col + np.arange(CB), 2] = 255
    col += CB

    # BM
    color_wheel[col + np.arange(BM), 2] = 255
    color_wheel[col + np.arange(BM), 0] = np.floor(255 * np.arange(BM) / BM)
    col += BM

    # MR
    color_wheel[col + np.arange(MR), 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    color_wheel[col + np.arange(MR), 0] = 255

    return color_wheel



def param_gen(k, mu, sigma, a, b, p):

    gamma = rn.gauss(mu, sigma)
    beta = np.random.binomial(1, p)
    return beta * max(min(np.sign(gamma) * abs(gamma)**k, b), a) + (1-beta)*mu

def warp_bg(j, w, h):

    r = rn.uniform(2, 3)
    bgw = int(w*r)
    bgh = int(h*r)

    X, Y = np.meshgrid(np.arange(0, bgw), np.arange(0, bgh))
    gr = np.dstack((X, Y))

    while True:

        tx = param_gen(4, 0, 1.3, -40, 40, 1)
        ty = param_gen(4, 0, 1.3, -40, 40, 1)
        an = param_gen(2, 0, 1.3, -10, 10, 0.3) # in degree
        sx = param_gen(2, 1, 0.1, 0.93, 1.07, 0.6)
        sy = param_gen(2, 1, 0.1, 0.93, 1.07, 0.6)

        shift_y, shift_x = rn.randint(0, h), rn.randint(0, w)
        tf_shift = SimilarityTransform(translation=[-shift_x, -shift_y])
        tf_shift_inv = SimilarityTransform(translation=[shift_x, shift_y])
        tf_rotate = tf_shift + (SimilarityTransform(rotation=np.deg2rad(an)) + tf_shift_inv)
        tf_trans  = SimilarityTransform(translation=[tx, ty])
        tf_scale  = AffineTransform(scale=(sx,sy))
        #form = AffineTransform(translation=(tx, ty), scale=(sx, sy), rotation=radians(an))
        gr_ = warp(gr.astype(np.float32), (tf_trans + tf_rotate + tf_scale).inverse,
            order=1, mode='constant', cval=-12345)

        fl = gr_ - gr

        # cropping from invalid regions
        # getting the available regions
        mask = np.logical_and(  gr_[...,0] > -12345/10,
                                gr_[...,1] > -12345/10,
                                np.logical_or(  gr_[...,0] != 0,
                                                gr_[...,0] != 0))
        csy = np.cumsum(mask, axis=0)
        csx = np.cumsum(mask, axis=1)
        idx = np.logical_and(csy > h+5, csx > w+5,
                             np.cumsum(csx, axis=0) > ((h+5)*(w+5)))
        ys, xs = np.where(idx==True)

        # get the pair of x, y that gives no contamination from warping
        for i in range(len(ys)):
            y, x = ys[i], xs[i]
            tmp = gr_[y-h:y,x-w:x,:]
            if (tmp < -12345/10).sum() == 0 and \
                np.all(tmp==(0.,0.),axis=2).sum() == 0:
                sy = y-h
                sx = x-w
                fl = fl[y-h:y,x-w:x,:]
                break

        if fl.min() > -100 and fl.max() < 100:
            break

    print j,
    sys.stdout.flush()
    Image.fromarray(convert(fl)).save('/hddstore/hale/bg/{:03d}.png'.format(j))

    return 1


def warp_bg1(w, h):
    # assuming that bg is landscape and already fit to image

    # scale the image to 2xlargest translation, then crop it back after warping
    #w, h = bg.size
    #r =  float(min(bg.size)+2*max_t) / min(bg.size)
    r = rn.uniform(2, 3)
    #bg = np.array(bg.resize((int(w*r), int(h*r)), Image.ANTIALIAS))
    bgw = int(w*r)
    bgh = int(h*r)

    X, Y = np.meshgrid(np.arange(0, bgw), np.arange(0, bgh))
    gr = np.dstack((X, Y))

    tx = param_gen(4, 0, 1.3, -40, 40, 1)
    ty = param_gen(4, 0, 1.3, -40, 40, 1)
    an = param_gen(2, 0, 1.3, -10, 10, 0.3) # in degree
    sx = param_gen(2, 1, 0.1, 0.93, 1.07, 0.6)
    sy = param_gen(2, 1, 0.1, 0.93, 1.07, 0.6)

    shift_y, shift_x = rn.randint(0, h), rn.randint(0, w)
    tf_shift = SimilarityTransform(translation=[-shift_x, -shift_y])
    tf_shift_inv = SimilarityTransform(translation=[shift_x, shift_y])
    tf_rotate = tf_shift + (SimilarityTransform(rotation=np.deg2rad(an)) + tf_shift_inv)
    tf_trans  = SimilarityTransform(translation=[tx, ty])
    tf_scale  = AffineTransform(scale=(sx,sy))
    #form = AffineTransform(translation=(tx, ty), scale=(sx, sy), rotation=radians(an))
    gr_ = warp(gr.astype(np.float32), (tf_trans + tf_rotate + tf_scale).inverse,
            order=3, mode='constant', cval=-12345)
    #bg_ = warp(bg.astype(np.float32), tform.inverse, order=3)[h/2:h/2+h, w/2:w/2+w,:]
    fl = (gr_ - gr)

    mask = np.logical_and(gr_[...,0] > -1234, gr_[...,1] > -1234)
    #maskc =  mask.sum(axis=0) >= h # showing which column can accommodate h
    #maskr =  mask.sum(axis=1) >= w # showing which row can accommodate w

    csy = np.cumsum(mask, axis=0)
    csx = np.cumsum(mask, axis=1)

    idx = np.logical_and(csy > h, csx > w, np.cumsum(csx, axis=0) > (h*w))
    ys, xs = np.where(idx==True)


    #try:
    #    x = min(np.where(np.cumsum(maskc) >= h)[0])
    #    y = min(np.where(np.cumsum(maskr) >= w)[0])
    #except:
    #    import pdb
    #    pdb.set_trace()

    #x  = min(np.where(maskc >= h)[0])
    #y  = min(np.where(maskr >= w)[0])

    #ph = maskc[x-h] # possible h
    #pw = maskr[y-w] # possible w

    for y, x in zip(ys, xs):
        fl = fl[y-h:y, x-w:x, :]
        if (fl[...,0] < -1234).sum() == 0 and (fl[...,1] < -1234).sum() == 0:
            break
            ##import pdb
            ##pdb.set_trace()
            ##plt.imshow(convert(fl)), plt.show()
            ##return 0
    else:
        import pdb
        pdb.set_trace()
        print 'Width: {:d} | Height: {:d} | x: {:d} | y: {:d} | Ratio: {:.4f} | tx: {:.4f} | ty: {:.4f} | an: {:.4f} | sx: {:.4f} | sy: {:.4f}'.format(w, h, x, y, r, tx, ty, an, sx, sy)
        return 0
    return 1


#        print counter
#        assert counter < 10, 'Failed!'
#
#    assert fl.shape[:-1] == bg_.shape[:-1], 'Warped BG and Flow has different size '\
#            'is {} vs. {}'.format(str(fl.shape[:-1]), str(bg_.shape[:-1]))
#
#    print 'Min: {:f} | Max: {:f}'.format(fl.min(), fl.max())
#    return fl, bg_.astype(np.uint8)


counter = 0
begin = time.time()
for i in range(999999999):
    if i % 100 == 0:
        print '\nStatus: i = {:d} | Counter = {:d} | Time per image = {:f}s'.format(i, counter, (time.time()-begin)/100)
        begin = time.time()
    w = rn.randint(800, 1500)
    h = rn.randint(500, w-100)
    counter += warp_bg(i, w, h)#1200, 800)


#plt.imshow(bg), plt.figure(), plt.imshow(convert(fl)), plt.figure(), plt.imshow(bg_), plt.show()
