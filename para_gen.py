import re, os, sys, time, numpy as np, random as rn, os.path as osp
from math import radians
from skimage.transform import warp, AffineTransform
import argparse, shutil, logging
from math import sqrt
from PIL import Image
from multiprocessing import Process, Queue
from subprocess import call
import sintel_io


arap_bin = '/home/hale/TrimBot/projects/ARAP_flow/Warp/deformation/image_warping'
dm_bin = 'deepmatching/deepmatching_1.2.2_c++/deepmatching-static'

input_root = 'data/DAVIS/'
output_root = 'data/DAVIS/fd1'
#bg_dir = '/home/hale/TrimBot/projects/flickr_downloader/img_downloads'
bg_dir = 'data/naturedata'

orgcolor = 'orgRGB'
orgmask = 'orgMasks'
color_dir = 'inpRGB'
mask_dir = 'inpMasks'
constraints_dir = 'tmpCnstr'

flow_dir = 'Flow'
wrgb_dir = 'wRGB'
wMask_dir = 'wMasks'

fd = 1

ARAP_BG = 255
WARP_CONST = -12345

parser = argparse.ArgumentParser(description='Arguments for ARAP flow generation')

#=============================================================================

def param_gen(k, mu, sigma, a, b, p):

    gamma = rn.gauss(mu, sigma)
    beta = np.random.binomial(1, p)
    return beta * max(min(np.sign(gamma) * abs(gamma)**k, b), a) + (1-beta)*mu


def warp_bg(bg, w, h):

    X, Y = np.meshgrid(np.arange(0, bg.shape[1]), np.arange(0, bg.shape[0]))
    X -= int(bg.shape[1] / 2)
    Y -= int(bg.shape[0] / 2)
    grid = np.dstack((X, Y))

    while True:

        tx = param_gen(4, 0, 1.3, -40, 40, 1)
        ty = param_gen(4, 0, 1.3, -40, 40, 1)
        an = param_gen(2, 0, 1.3, -10, 10, 0.3) # in degree
        sx = param_gen(2, 1, 0.1, 0.93, 1.07, 0.6)
        sy = param_gen(2, 1, 0.1, 0.93, 1.07, 0.6)

        tform = AffineTransform(translation=(tx, ty), scale=(sx, sy), rotation=radians(an))
        grid_ = warp(grid.astype(np.float32), tform.inverse, order=1, mode='constant', cval=WARP_CONST)
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
    r = rn.uniform(2, 2.5) * max(float(hmax)/bgh, float(wmax)/bgw)
    bgim = bgim.resize((int(bgw*r), int(bgh*r)), Image.ANTIALIAS)
    bg = np.array(bgim)

    if not static:
        bg, bg2, bgfl = warp_bg(bg, imw, imh)
    else:
        # random position to crop
        sy, sx = rn.randint(0, bg.shape[0] - imh), rn.randint(0, bg.shape[1] - imw)
        # cropping
        bg = bg[sy:(sy+imh), sx:(sx+imw), :]
        bg2 = bg
        bgfl = np.zeros((bg.shape[0], bg.shape[1], 2))

    return bg, bg2, bgfl


def fit_bg(bg, im, static=True):
    imh, imw = im.shape[:2]
    bgh, bgw = bg.shape[:2]
    bgim = Image.fromarray(bg)

    hmax = max(bgh, imh)
    wmax = max(bgw, imw)
    # scale the background to at least 2 times larger than the image
    r = rn.uniform(2, 2.5) * max(float(hmax)/bgh, float(wmax)/bgw)
    bgim = bgim.resize((int(bgw*r), int(bgh*r)), Image.ANTIALIAS)
    bg = np.array(bgim)

    if not static:
        bg, bg2, bgfl = warp_bg(bg, imw, imh)
    else:
        # random position to crop
        sy, sx = rn.randint(0, bg.shape[0] - imh), rn.randint(0, bg.shape[1] - imw)
        # cropping
        bg = bg[sy:(sy+imh), sx:(sx+imw), :]
        bg2 = bg
        bgfl = np.zeros((bg.shape[0], bg.shape[1], 2))

    return bg, bg2, bgfl

def add_bg(im, mk, bgim, bgval=0):
    assert mk.shape == im.shape[:-1], 'Sizes mismatch mask and image '+\
            str(mk.shape) + ' vs. ' + str(im.shape[:-1])
    assert bgim.shape == im.shape, 'Sizes mismatch background and image '+\
            str(bgim.shape) + ' vs. ' + str(im.shape)
    out = im.copy()
    idx = mk==bgval
    if len(out.shape) == 3:
        out[idx] = bgim[idx]
    else:
        out = bgim
    return out

def bg_gen(bg_dir, im1paths, im2paths, flow_root):

    # get list of all background
    bg_paths = []
    print "Scanning background directory... ",
    begin = time.time()
    for root, _, files in os.walk(bg_dir):
        for f in files:
            try:
                im = np.array(Image.open(osp.join(root, f)))
                if im.shape[2] == 3:
                    bg_paths.append(osp.join(root, f))
            except:
                continue
    print "\t[Done] | {:.2f} mins".format((time.time()-begin)/60)

    print "Adding backgrounds to frames...",
    begin = time.time()
    tmp_paths = []
    # scan by im2paths because frame2 might contains fewer images than original
    lines = []
    for root, _, files in os.walk(im2paths['rgb_root']):
        for f in files:
            if '.PNG' not in f.upper():
                continue
            # strip form the slashes
            p = root.replace(im2paths['rgb_root'], '').strip(osp.sep)
            ff = f.replace('.png', '.flo')

            assert osp.exists(osp.join(im1paths['rgb_root'], p, f)),\
                        'File not found ' + osp.join(im1paths['rgb_root'], p, f)
            assert osp.exists(osp.join(im1paths['mask_root'], p, f)),\
                        'File not found ' + osp.join(im1paths['mask_root'], p, f)
            assert osp.exists(osp.join(im2paths['rgb_root'], p, f)),\
                        'File not found ' + osp.join(im2paths['rgb_root'], p, f)
            assert osp.exists(osp.join(im2paths['mask_root'], p, f)),\
                        'File not found ' + osp.join(im2paths['mask_root'], p, f)
            assert osp.exists(osp.join(flow_root, p, ff)),\
                        'File not found ' + osp.join(flow_root, p, ff)

            im1 = np.array(Image.open(osp.join(im1paths['rgb_root'], p, f)))
            mk1 = np.array(Image.open(osp.join(im1paths['mask_root'], p, f)))
            im2 = np.array(Image.open(osp.join(im2paths['rgb_root'], p, f)))
            mk2 = np.array(Image.open(osp.join(im2paths['mask_root'], p, f)))

            # load background
            if len(tmp_paths) == 0:
                tmp_paths = sorted(bg_paths[:])
            bgpath = rn.choice(tmp_paths)
            tmp_paths.remove(bgpath)
            bgim = np.array(Image.open(bgpath))
            bgim = fit_bg(bgim, im1)

            # add background
            out1 = add_bg(im1, mk1, bgim, bgval=im1paths['bgval'])
            outpath1 = osp.join(im1paths['rgb_out'], p, f)
            if not osp.isdir(osp.dirname(outpath1)):
                os.makedirs(osp.dirname(outpath1))
            Image.fromarray(out1).save(outpath1)

            out2 = add_bg(im2, mk2, bgim, bgval=im2paths['bgval'])
            outpath2 = osp.join(im2paths['rgb_out'], p, f)
            if not osp.isdir(osp.dirname(outpath2)):
                os.makedirs(osp.dirname(outpath2))
            Image.fromarray(out2).save(outpath2)

            flowpath = osp.join(flow_root, p, ff)
            lines.append('\t'.join([osp.abspath(outpath1),
                                    osp.abspath(outpath2),
                                    osp.abspath(flowpath)]))
    print "\t[Done] | {:.2f} mins".format((time.time()-begin)/60)
    return lines

def flatten(arap_seg_paths):
    for arap_path, seg_paths in arap_seg_paths:
        assert len(seg_paths) > 0, 'Something wrong with seg_paths'
        _, msk1_path, _, flow_path, rgb2_path, msk2_path = seg_paths[0].split(' ')
        flow_im = np.dstack(sintel_io.flow_read(flow_path))
        rgb2_im = np.array(Image.open(rgb2_path))
        msk2_im = np.array(Image.open(msk2_path))

        if len(rgb2_im.shape) == 2:
            rgb2_im = rgb2_im[..., None]

        os.remove(flow_path)
        os.remove(rgb2_path)
        os.remove(msk2_path)

        for i in range(1, len(seg_paths)):
            _, msk1_path, _, flow_path, rgb2_path, msk2_path = seg_paths[i].split(' ')
            msk1_ob = np.array(Image.open(msk1_path)) == 0
            flow_ = np.dstack(sintel_io.flow_read(flow_path))
            rgb2_ = np.array(Image.open(rgb2_path))
            msk2_ = np.array(Image.open(msk2_path))
            msk2_ob = msk2_ != 0

            if len(rgb2_.shape) == 2:
                rgb2_ = np.dstack((rgb2_,rgb2_,rgb2_))

            flow_im[msk1_ob] = flow_[msk1_ob]
            rgb2_im[msk2_ob] = rgb2_[msk2_ob]
            msk2_im[msk2_ob] = msk2_ob[msk2_ob]

            os.remove(flow_path)
            os.remove(rgb2_path)
            os.remove(msk2_path)

        # output
        sintel_io.flow_write(arap_path.split(' ')[-3], flow_im)
        Image.fromarray(rgb2_im).save(arap_path.split(' ')[-2])
        Image.fromarray(msk2_im.astype(np.uint8)).save(arap_path.split(' ')[-1])

    return [entry[0] for entry in arap_seg_paths]


def do_arap(paths, gpu, gpu_queue, arap_seg_paths=[], bgs=[]):

    # create temporary list file to input to ARAP
    if not osp.isdir('tmp'):
        os.makedirs('tmp')
    fn = 'gpu-{:d}_{}'.format(gpu, str(time.time()).replace('.', '_'))
    print 'GPU ' , gpu , ' ' , len(paths) , ' files'
    try:
        open('tmp/{}.txt'.format(fn), 'w').write('\n'.join(paths))
        path = osp.abspath('tmp/{}.txt'.format(fn))

        # run arap from command line
        cmd = 'export CUDA_VISIBLE_DEVICES={:d} ; {} {}'.format(gpu, flags.arap_bin, path)

        #begin = time.time()
        status = call(cmd, shell=True)
        assert status == 0, \
            'ARAP exited with code {:d}. The command was \n{}'.format(status, cmd)
        #print '[{:.2f}% ] | Elapsed {:.3f}s'.format(progress*100, time.time() - begin)
        print 'Finish run_arap', path.split(' ')[0]
    finally:
        # clean up
        os.remove('tmp/{}.txt'.format(fn))

    # flatten all layers
    if len(arap_seg_paths) > 0:
        paths = flatten(arap_seg_paths)

    # add background
    # if len(bgs) == 0, the loop would exit immediately all by itself
    for path, bg in zip(paths, bgs):
        fl, pt, mk = path.split(' ')[-3:]
        im = np.array(Image.open(pt))
        mk = np.array(Image.open(mk))
        im = add_bg(im, mk, bg[0])
        Image.fromarray(im).save(pt)
        flo = np.dstack(sintel_io.flow_read(fl))
        flo = add_bg(flo, bg[2], bg[1])
        sintel_io.flow_write(fl, flo)

    # free the gpu
    gpu_queue.put(gpu)

def add_bg2(ps, bgim1, bgim2, bgflo, bgval=0):

    rgb1 = ps[0]
    msk1 = ps[1]
    flow = np.dstack(sintel_io.flow_read(ps[2]['flow_gen']))
    rgb2 = np.array(Image.open(ps[2]['rgb2_gen']))
    msk2 = np.array(Image.open(ps[2]['msk2_gen']))

    # calculating random displacement to shift the object
    idx = msk1 != bgval
    rows = np.any(idx, axis=1)
    cols = np.any(idx, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    ravg = (rmax + rmin) / 2
    cavg = (cmax + cmin) / 2

    rdisp = rn.randint(0, msk1.shape[0]) - ravg
    cdisp = rn.randint(0, msk1.shape[1]) - cavg

    # by padding the mask with background value, we shift the object
    pad = [ [(rdisp>0)*abs(rdisp),(rdisp<0)*abs(rdisp)],
            [(cdisp>0)*abs(cdisp),(cdisp<0)*abs(cdisp)]]

    h, w = msk1.shape
    r, c = (rdisp<0)*abs(rdisp), (cdisp<0)*abs(cdisp)
    msk1 = np.pad(msk1, pad, mode='constant', constant_values=bgval)[r:r+h,c:c+w]
    msk2 = np.pad(msk2, pad, mode='constant', constant_values=bgval)[r:r+h,c:c+w]
    rgb1 = np.pad(rgb1, pad+[(0,0)], mode='constant', constant_values=bgval)[r:r+h,c:c+w,:]
    flow = np.pad(flow, pad+[(0,0)], mode='constant', constant_values=bgval)[r:r+h,c:c+w,:]
    rgb2 = np.pad(rgb2, pad+[(0,0)], mode='constant', constant_values=bgval)[r:r+h,c:c+w,:]

    rgb1 = add_bg(rgb1, msk1, bgim1)
    flow = add_bg(flow, msk1, bgflo)
    rgb2 = add_bg(rgb2, msk2, bgim2)

    return rgb1, rgb2, flow

def valid_cnstr(x1, y1, x2, y2, msk1, msk2):

    if x1 >= msk1.shape[1] or x2 >= msk2.shape[1] or \
            y1 >= msk1.shape[0] or y2 >= msk2.shape[0]:
        return False

    dist = sqrt((x2-x1)**2 + (y2-y1)**2)
    return  dist < 60 and dist > 0 and msk1[y1, x1] > 0 and msk1[y1, x1] == msk2[y2, x2]


# TODO: have a mechanism to input constraints from file and not run again
def run_matching(img1, img2, msk1, msk2, out_file):

    assert osp.exists(img1), 'File not found: \n{}'.format(img1)
    assert osp.exists(img2), 'File not found: \n{}'.format(img2)
    assert osp.exists(msk1), 'File not found: \n{}'.format(msk1)
    assert osp.exists(msk2), 'File not found: \n{}'.format(msk2)

    cmd = './{} {} {} -nt 0 -out {} -ngh_rad 100 '.format(flags.dm_bin, img1, img2, out_file)
    # call the deep matching module from shell
    status = call(cmd, shell=True)
    assert status == 0, \
        'Deep matching exited with code {:d}. The command is \n{}'.format(status, cmd)

    print 'Done matching for ' + out_file


def has_mask(msk1_path, msk2_path):

    try:
        mask1 = np.array(Image.open(msk1_path))
        mask2 = np.array(Image.open(msk2_path))
    except:
        return False

    return mask1.sum() > 10 and mask2.sum() > 10

def scale_rotate(im_path, mk_path):

    im = Image.open(im_path)
    mk = Image.open(mk_path)

    assert im.size == mk.size, 'Image and mask must be of the same size' \
            'but given {:s} vs. {:s}'.format(str(im.size), str(mk.size))

    ext = '{} {}'.format(osp.splitext(im_path)[1], osp.splitext(mk_path)[1])

    preprocessed = False
    if 'JPG' in ext.upper() or 'JPEG' in ext.upper():
        preprocessed = True

    # check if the image is portrait, i.e. height > width
    if im.size[1] > im.size[0]:
        im = im.transpose(Image.TRANSPOSE)
        mk = mk.transpose(Image.TRANSPOSE)
        preprocessed = True

    # make all images to the same size. TODO: passed in as argument
    if flags.size is None:
        flags.size = im.size
    if im.size != flags.size:
        r = max(float(flags.size[0]+10) / float(im.size[0]),
                float(flags.size[1]+10)/ float(im.size[1]))
        w, h = (np.array(im.size) * r).astype(np.int)
        im = im.resize((w, h), Image.ANTIALIAS)
        mk = mk.resize((w, h), Image.NEAREST)

        left  = int(w/2) - flags.size[0]/2
        upper = int(h/2) - flags.size[1]/2
        right = left + flags.size[0]
        lower = upper + flags.size[1]

        im = im.crop((left, upper, right, lower))
        mk = mk.crop((left, upper, right, lower))

        preprocessed = True

    return (preprocessed, im, mk)


def preprocess(p):

    preprocessed, im1, mk1 = scale_rotate(p['rgb1_org'], p['msk1_org'])
    if preprocessed:
        im1.save(p['rgb1_gen'])
        mk1.save(p['msk1_gen'])
        p['rgb1_org'] = p['rgb1_gen']
        p['msk1_org'] = p['msk1_gen']

    preprocessed, im2, mk2 = scale_rotate(p['rgb2_org'], p['msk2_org'])
    if preprocessed:
        im2.save(p['rgb2_gen'])
        mk2.save(p['msk2_gen'])
        p['rgb2_org'] = p['rgb2_gen']
        p['msk2_org'] = p['msk2_gen']

    return np.array(im1), np.array(mk1), np.array(im2), np.array(mk2)

def cleanup(p):
    logging.info('Cleaning up')
    for k in p:
        if '_org' not in k and osp.exists(p[k]):
            logging.warning('Removing\n\t{}'.format(p[k]))
            os.remove(p[k])

def replace_ext(dict_path, seg_num, keep_orgs=[]):

    dict_out = dict()
    for k in dict_path:
        fn, ext = osp.splitext(dict_path[k])
        if k not in keep_orgs:
            dict_out[k] = fn + '_seg{:d}{:s}'.format(seg_num, ext)
        else:
            dict_out[k] = dict_path[k]

    return dict_out

def make_arap_path(p):

    arap_path =' '.join([osp.abspath(p['rgb1_gen']),
                        osp.abspath(p['msk1_gen']),
                        osp.abspath(p['cstr_tmp']),
                        osp.abspath(p['flow_gen']),
                        osp.abspath(p['rgb2_gen']),
                        osp.abspath(p['msk2_gen'])])
    return arap_path

def generic_pipeline():#num, objs, root, rgb_org, msk_org, cst_root, flo_root, rgb_root, msk_root, wco_root, wmk_root):
    '''
        objs:   list of DAVIS sequences, within each sequence must be the images
        root:   path to where all the sequences are
    '''

    num = 5
    root = 'data/DAVIS/orgRGB'
    objs = os.listdir(root)

    # TODO have the file pattern input from argument
    reg = re.compile('(\d+)\.jp.?g', flags=re.IGNORECASE) # or put (?i)jp.g

    tmp_paths = []
    bg_paths = []
    print "Scanning background directory... ",
    begin = time.time()
    for bgroot, _, files in os.walk(bg_dir):
        for f in files:
            if '.PNG' not in f.upper() and \
                '.JPG' not in f.upper() and  '.JPEG' not in f.upper():
                continue
            bg_paths.append(osp.join(bgroot, f))
    print "\t[Done] | {:.2f} mins".format((time.time()-begin)/60)

    # for each image in the whole dataset
    for iframe in range(num):
        nobjs = 3 # np.randint(6, 10)
        pickeds = []


        # load background
        while True:
            if len(tmp_paths) == 0:
                tmp_paths = sorted(bg_paths[:]) # copy
            bgpath = rn.choice(tmp_paths)
            tmp_paths.remove(bgpath)
            try:
                bgim = np.array(Image.open(bgpath))
                if bgim.shape[2] == 3:
                    break
            except:
                pass
            # if something wrong happens
            bg_paths.remove(bgpath)

        # fit background to the image size
        bgim, bgim2, bgflo = prepare_bg(bgim, flags.size[::-1])


        lmdb_paths = []
        arap_paths = []
        arap_seg_paths = []

        ngpus = len(flags.gpu)
        gpu_queue = Queue(ngpus)
        for g in flags.gpu:
            gpu_queue.put(g)
        procs = {}
        path_segments = [] # keeping track of path set for each object for bg pasting
        # for each object in n objects to be put in this image
        for iobj in range(nobjs):

            # sampling an object without replacement
            while True:
                seq = rn.choice(objs)
                if seq not in pickeds:
                    pickeds.append(seq)
                    break

            # sampling a frame distance
            fd = rn.randint(1, 5) # TODO: incorporate into output_root

            # randomly pick a file
            files = sorted([f for f in os.listdir(osp.join(root, seq))
                        if reg.search(f) is not None])

            while True:
                f1 = rn.choice(files[:-fd-1])
                f, ext = osp.splitext(f1) # strip extension path
                # getting frame number
                num = reg.search(f1)
                if num is None:
                    continue
                n = '{:0'+str(len(num.group(1)))+'d}'
                # getting next frame according to frame distance fd
                n = n.format(int(num.group(1))+fd)
                f2 = f.replace(num.group(1), n)

                if f2+ext in files:
                    break

                ## TODO sanity check if all required files exist for the pair of chosen frames
                #if not osp.exists(osp.join(msk_org, seq, f +'.png')):
                #    continue

            ## skipping if out of second frame
            #if not osp.exists(osp.join(rgb_org, seq, f2+ext)) or \
            #    not osp.exists(osp.join(msk_org, seq, f2+'.png')):
            #    continue


            fdn = 'fd{:d}'.format(fd)
            rgb_org = osp.join(input_root, orgcolor)
            msk_org = osp.join(input_root, orgmask)
            cst_root = osp.join(output_root, fdn, constraints_dir)
            flo_root = osp.join(output_root, fdn, flow_dir)

            rgb_root = osp.join(output_root, fdn, color_dir)
            msk_root = osp.join(output_root, fdn, mask_dir)
            wco_root = osp.join(output_root, fdn, wrgb_dir)
            wmk_root = osp.join(output_root, fdn, wMask_dir)



            # generating arap path set
            entry = {}
            entry['rgb1_gen'] = osp.abspath(osp.join(rgb_root, seq, f+'.png'))
            entry['msk1_gen'] = osp.abspath(osp.join(msk_root, seq, f+'.png'))
            entry['rgb2_gen'] = osp.abspath(osp.join(wco_root, seq, f+'.png'))
            entry['msk2_gen'] = osp.abspath(osp.join(wmk_root, seq, f+'.png'))

            entry['cstr_tmp'] = osp.abspath(osp.join(cst_root, seq, f+'.txt'))
            entry['flow_gen'] = osp.abspath(osp.join(flo_root, seq, f+'.flo'))

            entry['rgb1_org'] = osp.abspath(osp.join(rgb_org, seq, f1))
            entry['msk1_org'] = osp.abspath(osp.join(msk_org, seq, f+'.png'))
            entry['rgb2_org'] = osp.abspath(osp.join(rgb_org, seq, f2+ext))
            entry['msk2_org'] = osp.abspath(osp.join(msk_org, seq, f2+'.png'))

            for k in entry:
                if not osp.exists(entry[k]):
                    break
            else:
                # loading data from entry
                # pasting onto to the background # TODO
                # saving to file
                continue

            # if not exists
            im1, mk1 = run_1image(entry, lmdb_paths, arap_paths, arap_seg_paths)

            # we dont need this: we dont paste segments onto background in this stage
            #bgim = add_bg(im1, mk1, bgim) # TODO paste at random position
            path_segments.append((im1, mk1, entry))
            # output to file
            #bgs.append((bgim2, bgflo, mk1))
            #Image.fromarray(out1).save(p['rgb1_gen'])


            #do_arap(arap_paths, 0, arap_seg_paths, [])
            #arap_paths = []
            #arap_seg_paths = []

            if not gpu_queue.empty():
                gpu = gpu_queue.get()
                proc = Process(target=do_arap,
                            args=(arap_paths, gpu, gpu_queue, arap_seg_paths, []))
                proc.start()
                procs[gpu] = proc
                arap_paths = []
                arap_seg_paths = []

        # wait for all the threads to finish
        if len(procs) > 0:
            for k in procs:
                procs[k].join()

        for ps in path_segments:
            bgim, bgim2, bgflo = add_bg2(ps, bgim, bgim2, bgflo)

        outpath = osp.join(output_root, 'data')
        if not osp.isdir(outpath):
            os.makedirs(outpath)
        Image.fromarray(bgim).save(osp.join(outpath, '{:05d}_1.png'.format(iframe)))
        Image.fromarray(bgim2).save(osp.join(outpath, '{:05d}_2.png'.format(iframe)))
        sintel_io.flow_write(osp.join(outpath, '{:05d}_f.flo'.format(iframe)),bgflo)

def main():

    rgb_org = osp.join(input_root, orgcolor)
    msk_org = osp.join(input_root, orgmask)
    cst_root = osp.join(output_root, constraints_dir)
    flo_root = osp.join(output_root, flow_dir)

    rgb_root = osp.join(output_root, color_dir)
    msk_root = osp.join(output_root, mask_dir)
    wco_root = osp.join(output_root, wrgb_dir)
    wmk_root = osp.join(output_root, wMask_dir)

    im1paths = dict()
    im1paths['rgb_root'] = osp.join(input_root, color_dir)
    im1paths['mask_root'] = osp.join(input_root, mask_dir)
    im1paths['rgb_out'] = osp.join(output_root, color_dir)
    im1paths['bgval'] = ARAP_BG

    im2paths = dict()
    im2paths['rgb_root'] = osp.join(output_root, wrgb_dir)
    im2paths['mask_root'] = osp.join(output_root, wMask_dir)
    im2paths['rgb_out'] = osp.join(output_root, wrgb_dir)
    im2paths['bgval'] = 0

    # get list of all background
    bg_paths = []
    print "Scanning background directory... ",
    begin = time.time()
    for root, _, files in os.walk(bg_dir):
        for f in files:
            if '.PNG' not in f.upper() and \
                '.JPG' not in f.upper() and  '.JPEG' not in f.upper():
                continue
            bg_paths.append(osp.join(root, f))
    print "\t[Done] | {:.2f} mins".format((time.time()-begin)/60)

    tmp_paths = []

    begin = time.time()

    bgs = []
    all_paths = []
    # TODO have the file pattern input from argument
    reg = re.compile('(\d+)\.jp.?g', flags=re.IGNORECASE) # or put (?i)jp.g

    print 'Scanning data to be processed',
    sys.stdout.flush()
    begin = time.time()
    for root, dirs, _ in os.walk(rgb_org):
        # check if the folder contain files of the wanted pattern
        for d in dirs:
            files = [f for f in os.listdir(osp.join(root, d))
                        if reg.search(f) is not None]
            for f1 in files:

                seq = root.replace(rgb_org, '').strip(osp.sep)
                seq = osp.join(seq, d)
                f, ext = osp.splitext(f1) # strip extension path

                if not osp.exists(osp.join(msk_org, seq, f +'.png')):
                    continue

                # getting frame number
                num = reg.search(f1)
                if num is None:
                    continue
                n = '{:0'+str(len(num.group(1)))+'d}'

                # getting next frame according to frame distance fd
                nxt = int(num.group(1))+flags.fd
                f2 = f.replace(num.group(1), n.format(nxt))
                # skipping if out of second frame
                if not osp.exists(osp.join(rgb_org, seq, f2+ext)) or \
                    not osp.exists(osp.join(msk_org, seq, f2+'.png')):
                    continue

                entry = {}
                entry['rgb1_gen'] = osp.abspath(osp.join(rgb_root, seq, f+'.png'))
                entry['msk1_gen'] = osp.abspath(osp.join(msk_root, seq, f+'.png'))
                entry['rgb2_gen'] = osp.abspath(osp.join(wco_root, seq, f+'.png'))
                entry['msk2_gen'] = osp.abspath(osp.join(wmk_root, seq, f+'.png'))

                entry['cstr_tmp'] = osp.abspath(osp.join(cst_root, seq, f+'.txt'))
                entry['flow_gen'] = osp.abspath(osp.join(flo_root, seq, f+'.flo'))

                entry['rgb1_org'] = osp.abspath(osp.join(rgb_org, seq, f1))
                entry['msk1_org'] = osp.abspath(osp.join(msk_org, seq, f+'.png'))
                entry['rgb2_org'] = osp.abspath(osp.join(rgb_org, seq, f2+ext))
                entry['msk2_org'] = osp.abspath(osp.join(msk_org, seq, f2+'.png'))

                if not flags.resume or not osp.exists(entry['flow_gen']):
                    all_paths.append(entry)

    print '\t\t{:d} files [Done] | {:.3f} seconds'.format(len(all_paths), time.time() - begin)

    #all_paths = all_paths[:10]

    lmdb_paths = []
    arap_paths = []
    arap_seg_paths = []
    procs = {}
    ngpus = len(flags.gpu)
    gpu_queue = Queue(ngpus)
    for g in flags.gpu:
        gpu_queue.put(g)
    for i, p in enumerate(all_paths):
        print '{:.3f}%'.format(float(i) * 100 / len(all_paths))

        # load background
        while True:
            if len(tmp_paths) == 0:
                tmp_paths = sorted(bg_paths[:]) # copy
            bgpath = rn.choice(tmp_paths)
            tmp_paths.remove(bgpath)
            try:
                bgim = np.array(Image.open(bgpath))
                if bgim.shape[2] == 3:
                    break
            except:
                pass
            # if something wrong happens
            bg_paths.remove(bgpath)

        # fit background to the image size
        bgim, bgim2, bgflo = prepare_bg(bgim, flags.size[::-1])

        im1, mk1 = run_1image(p, lmdb_paths, arap_paths, arap_seg_paths)

        out1 = add_bg(im1, mk1, bgim)
        # output to file
        bgs.append((bgim2, bgflo, mk1))
        Image.fromarray(out1).save(p['rgb1_gen'])


        #do_arap(arap_paths, 0, arap_seg_paths, bgs)
        #arap_paths = []
        #arap_seg_paths = []
        #bgs = []

        if not gpu_queue.empty():
            gpu = gpu_queue.get()
            proc = Process(target=do_arap,
                        args=(arap_paths, gpu, gpu_queue, arap_seg_paths, bgs))
            proc.start()
            procs[gpu] = proc
            arap_paths = []
            arap_seg_paths = []
            bgs = []

    #if len(arap_paths) > flags.narap:
    #    if procs is not None:
    #        for proc in procs:
    #            proc.join()
    #    npaths = len(arap_paths)
    #    ngpus = len(flags.gpu)
    #    d = npaths // ngpus
    #    if npaths % ngpus > 0:
    #        d += 1
    #    for i, gpu in enumerate(flags.gpu):
    #        paths = arap_paths[(i*d):min((i+1)*d, npaths)]
    #        proc = Process(target=do_arap, args=(paths, bgs, gpu))
    #        proc.start()
    #        procs.append(proc)
    #    arap_paths = []
    #    bgs = []



        # add background

    # wait for all the threads to finish
    if len(procs) > 0:
        for k in procs:
            procs[k].join()

    # check sums
    out_paths = []
    for line in lmdb_paths:
        all_good = True
        for l in line.split(' '):
            if not osp.exists(l):
                all_good = False
                break
        if all_good:
            out_paths.append(line)
    open(osp.join(output_root, 'all_files.list'), 'w').write('\n'.join(out_paths))
    return out_paths
    shutil.rmtree('tmp')


def run_1image(p, lmdb_paths, arap_paths, arap_seg_paths):

    arap_path = make_arap_path(p)
    lmdb_paths.append(' '.join([arap_path.split(' ')[l] for l in [0, 4, 3]]))

    # preparing for output
    for k in p:
        if not osp.isdir(osp.dirname(p[k])):
            os.makedirs(osp.dirname(p[k]))

    im1, mk1, im2, mk2 = preprocess(p)

    if not has_mask(p['msk1_org'], p['msk2_org']):
        cleanup(p)
        return

    run_matching(p['rgb1_org'], p['rgb2_org'],
                p['msk1_org'], p['msk2_org'], p['cstr_tmp'])

    # Filter constraints to  belong in the same segment type and within radius 60px
    # Load mask1 and mask2 images
    cstr_lines = open(p['cstr_tmp']).read().splitlines()
    cstrs = []
    valids = [] # valid segment number in case of multiple segment
    # check the constraints
    for line in cstr_lines:
        x1, y1, x2, y2 = [int(l) for l in line.split(' ')[:4]]

        if valid_cnstr(x1, y1, x2, y2, mk1, mk2):
            cstrs.append('\t'.join(['{:d}']*4).format(x1, y1, x2, y2))
            valids.append(mk1[y1, x1])
    # write back to file
    open(p['cstr_tmp'], 'w').write('\n'.join([str(len(cstrs))] + cstrs))
    if len(cstrs) == 0:
        cleanup(p)
        return

    # Convert mask
    if not osp.isdir(osp.dirname(p['msk1_gen'])):
        os.makedirs(osp.dirname(p['msk1_gen']))

    seg_paths = None
    if not flags.multseg:
        mask = np.zeros_like(mk1, dtype=np.uint8)
        mask[mk1==0] = ARAP_BG
        Image.fromarray(mask).save(p['msk1_gen'])
    else:
        seg_paths = []
        for s in np.unique(valids): # only check the valid segment (segment with at least 1 constraint)
            if s == 0:
                continue
            p_ = replace_ext(p, s, keep_orgs=['rgb1_gen', 'cstr_tmp'])

            # output mask per segment
            mask = np.zeros_like(mk1, dtype=np.uint8) + ARAP_BG
            mask[mk1 == s] = 0
            Image.fromarray(mask).save(p_['msk1_gen'])

            # output constraints per segment
            cstr_segs = []
            for line in cstr_lines:
                x1, y1, x2, y2 = [int(l) for l in line.split(' ')[:4]]
                if mk1[y1, x1] == s:
                    cstr_segs.append('\t'.join(['{:d}']*4).format(x1, y1, x2, y2))
            assert len(cstr_segs) > 0, 'Segment {:s} has no constraint'.format(s)

            # output constraints per segment
            seg_paths.append(make_arap_path(p_))
        arap_seg_paths.append((arap_path, seg_paths))


    for sp in arap_path.split(' ')[:2]:
        assert osp.exists(sp), 'File not found:\n{}'.format(sp)
    for sp in arap_path.split(' ')[3:]:
        if not osp.isdir(osp.dirname(sp)):
            os.makedirs(osp.dirname(sp))

    if seg_paths is None:
        arap_paths.append(arap_path)
    else:
        arap_paths += seg_paths

    return im1, mk1




if __name__ == "__main__":
    # TODO check if rgb and mask images are in png format, if not convert them to png
    # TODO input if want to keep constraints and warped mask
    parser.add_argument('--input', type=str, required=True,
            help='Path to input root')
    parser.add_argument('--output', type=str, required=True,
            help='Path to output root')
    parser.add_argument('--rm-cnstr')
    parser.add_argument('--rm-wmask')
    parser.add_argument('--rm-tmp-cmd')
    parser.add_argument('--img-pattern')
    parser.add_argument('--gpu', nargs='*', type=int, default=[0],
            help='GPU id to be used, default=0')
    parser.add_argument('--multseg', action='store_true', default=False,
            help='if each object segment is treated separately')
    parser.add_argument('--resume', action='store_true', default=False,
            help='To skip the images that have *.flo finished.')
    parser.add_argument('--narap', type=int, default=7,
            help='Number of buffered files to be run by ARAP on gpu; should be '
            'balanced with deep matching running in CPU; default=7')
    parser.add_argument('--size', nargs=2, default=None,
            help='2-tuple of [width] [space] [height] (in this order) '
            'to which all images are resized. '
            'Omit to keep original dimensions of images.')
    parser.add_argument('--fd', type=int, default=1,
            help='Positive integer indicating the distance between 2 frames to '
            'generate optical flow; should not be larger than 10 for better '
            'results, default=1 for consecutive frames')
    parser.add_argument('--arap_bin', default='./arap-deform',
            help='Path to built ARAP binary file to be run, default=./arap-deform')
    parser.add_argument('--dm_bin', default='./dm',
            help='Path to built deep matching binary file to be run, default=./dm')
    flags = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(flags.gpu)
    if flags.size is not None:
        flags.size = tuple([int(s) for s in flags.size])
    assert flags.fd > 0 and flags.fd < 20, 'Invalid fd number!'
    assert osp.exists(flags.arap_bin), 'File not found ' + flags.arap_bin
    assert osp.exists(flags.dm_bin), 'File not found ' + flags.dm_bin

    input_root = flags.input.rstrip(osp.sep)
    output_root = flags.output.rstrip(osp.sep)

    logging.basicConfig(filename='example.log',level=logging.DEBUG)
    #main()
    generic_pipeline()
