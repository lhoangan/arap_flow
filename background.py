from math import pi, radians
import numpy as np, random as rn
from skimage.transform import warp, AffineTransform
import matplotlib.pyplot as plt
from PIL import Image

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

def warp_bg(bg):
    # assuming that bg is landscape and already fit to image

    # scale the image to 2xlargest translation, then crop it back after warping
    w, h = bg.size
    #r =  float(min(bg.size)+2*max_t) / min(bg.size)
    r = 2
    bg = np.array(bg.resize((int(w*r), int(h*r)), Image.ANTIALIAS))

    X, Y = np.meshgrid(np.arange(0, bg.shape[1]), np.arange(0, bg.shape[0]))
    gr = np.dstack((X, Y))


    counter = 0
    while (True):
        tx = param_gen(4, 0, 1.3, -40, 40, 1)
        ty = param_gen(4, 0, 1.3, -40, 40, 1)
        an = param_gen(2, 0, 1.3, -10, 10, 0.3) # in degree
        sx = param_gen(2, 1, 0.1, 0.93, 1.07, 0.6)
        sy = param_gen(2, 1, 0.1, 0.93, 1.07, 0.6)

        tform = AffineTransform(translation=(tx, ty), scale=(sx, sy), rotation=radians(an))
        gr_ = warp(gr.astype(np.float32), tform.inverse, order=3, mode='constant', cval=-12345)
        bg_ = warp(bg.astype(np.float32), tform.inverse, order=3)[h/2:h/2+h, w/2:w/2+w,:]
        fl = (gr_ - gr)[h/2:h/2+h,w/2:w/2+w,:]

        if (fl < -1234).sum() == 0:
            break
        else:
            plt.imshow(convert(fl)), plt.show()
            import pdb
            pdb.set_trace()
            counter += 1
        print counter
        assert counter < 10, 'Failed!'

    assert fl.shape[:-1] == bg_.shape[:-1], 'Warped BG and Flow has different size '\
            'is {} vs. {}'.format(str(fl.shape[:-1]), str(bg_.shape[:-1]))

    print 'Min: {:f} | Max: {:f}'.format(fl.min(), fl.max())
    return fl, bg_.astype(np.uint8)

bg = Image.open('/hddstore/hale/naturedata/nature/JPEGImages_4/selfie/90.png')
fl, bg_ = warp_bg(bg)

plt.imshow(bg), plt.figure(), plt.imshow(convert(fl)), plt.figure(), plt.imshow(bg_), plt.show()
