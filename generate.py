import sys, re, time
import argparse
import numpy as np
import random as rn
from math import sqrt
from PIL import Image
import os, os.path as osp
import multiprocessing
from subprocess import call
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

arap_bin = '/home/hale/TrimBot/projects/ARAP_flow/Warp/deformation/image_warping'
dm_bin = 'deepmatching/deepmatching_1.2.2_c++/deepmatching-static'

input_root = 'data/DAVIS/'
output_root = 'data/DAVIS/fd1'
bg_dir = '/home/hale/TrimBot/projects/flickr_downloader/tt'

orgcolor = 'orgRGB'
orgmask = 'orgMasks'
color_dir = 'inpRGB'
mask_dir = 'inpMasks'
constraints_dir = 'tmpCnstr'

flow_dir = 'Flow'
wrgb_dir = 'wRGB'
wMask_dir = 'wMasks'

frame_distance = 1

ARAP_BG = 255


#=============================================================================

def fit_bg(bg, im):
    imh, imw, _ = im.shape
    bgh, bgw, _ = bg.shape
    bgim = Image.fromarray(bg)

    hmax = max(bgh, imh)
    wmax = max(bgw, imw)
    r = rn.uniform(1, 2) * max(float(hmax)/bgh, float(wmax)/bgw)
    bgim = bgim.resize((int(bgw*r), int(bgh*r)), Image.ANTIALIAS)
    bg = np.array(bgim)
    # random crop the background to the image
    sy, sx = rn.randint(0, bg.shape[0] - imh), rn.randint(0, bg.shape[1] - imw)
    return bg[sy:(sy+imh), sx:(sx+imw), :]

def add_bg(im, mk, bgim, bgval):
    assert mk.shape == im.shape[:-1], 'Sizes mismatch mask and image '+\
            str(mk.shape) + ' vs. ' + str(im.shape[:-1])
    assert bgim.shape == im.shape, 'Sizes mismatch background and image '+\
            str(bgim.shape) + ' vs. ' + str(im.shape)
    out = im.copy()
    idx = mk==bgval
    out[idx] = bgim[idx]
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
            lines.append('\t'.join([outpath1, outpath2, flowpath ]))
    print "\t[Done] | {:.2f} mins".format((time.time()-begin)/60)
    return lines


def run_arap(path, progress):
    cmd = '{} {:s}'.format(arap_bin, path)

    begin = time.time()
    status = call(cmd, shell=True)
    assert status == 0, \
        'ARAP exited with code {:d}. The command was \n{}'.format(status, cmd)
    print '[{:.2f}% ] | Elapsed {:.3f}s'.format(progress*100, time.time() - begin)

# TODO strip all the input path trailing slash
def arap_deform(rgb_root, msk_root, cst_root, flo_root, wco_root, wmk_root):

    paths = dict()
    # scan by constraints files since rgb_root contains more file than there really is
    scan_root = cst_root
    for root, dirs, files in os.walk(scan_root):
        for f in files:
            if '.txt' not in f:
                continue
            seq = root.replace(scan_root, '').strip(osp.sep) # strip form the slashes
            paths[seq] = paths.get(seq, [])
            fi = f.replace('.txt','.png') # filename for img
            ff = f.replace('.txt','.flo') # filename for flow

            line = '{} {} {} {} {} {}'.format(  osp.join(rgb_root, seq, fi),
                                                osp.join(msk_root, seq, fi),
                                                osp.join(cst_root, seq, f),
                                                osp.join(flo_root, seq, ff),
                                                osp.join(wco_root, seq, fi),
                                                osp.join(wmk_root, seq, fi))
            for p in line.split(' ')[:2]:
                assert osp.exists(p), 'File not found:\n{}'.format(p)
            for p in line.split(' ')[3:]:
                if not osp.isdir(osp.dirname(p)):
                    os.makedirs(osp.dirname(p))
            paths[seq].append(line)

    # create temporary list file to input to ARAP
    if not osp.isdir('tmp'):
        os.makedirs('tmp')
    files = []
    # create temporary list files
    for key in paths:
        # TODO use try and finally to clean the temporary files
        fn = key.replace('/', '_')
        # TODO tmp folder
        open('tmp/{}.txt'.format(fn), 'w').write('\n'.join(paths[key]))
        files.append(osp.abspath('tmp/{}.txt'.format(fn)))

    num_cores = int(multiprocessing.cpu_count() / 2) # TODO gpu capcity
    n = float(len(paths.keys()))
    Parallel(n_jobs=num_cores)(delayed(run_arap)(p, i/n) for i, p in enumerate(files))

def convert_rgb(jpg_root, png_root):
    for root, _, files in os.walk(jpg_root):
        for f in files:
            if '.JPG' not in f.upper() and '.JPEG' not in f.upper():
                continue
            im = Image.open(osp.join(root, f))
            outdir = root.replace(jpg_root, png_root)
            if not osp.isdir(outdir):
                os.makedirs(outdir)
            im.save(osp.join(outdir, osp.splitext(f)[0]+'.png'))

def convert_mask(inp_root, out_root, flag=None):
    """
        inp_root: root to instant masks, background = 0, objects > 0
        out_root: root to output masks, object = 0, background > 0
    """
    for root, _, files in os.walk(inp_root):
        for f in files:
            im = np.array(Image.open(osp.join(root, f)))
            outdir = root.replace(inp_root, out_root)
            if not osp.isdir(outdir):
                os.makedirs(outdir)
            mask = np.zeros_like(im, dtype=np.uint8)
            mask[im==0] = ARAP_BG # TODO mask with each object segment separately
            Image.fromarray(mask).save(osp.join(outdir, osp.splitext(f)[0]+'.png'))

# TODO: have a mechanism to input constraints from file and not run again
def run_matching(img1, img2, msk1, msk2, out_file):

    assert osp.exists(img1), 'File not found: \n{}'.format(img1)
    assert osp.exists(img2), 'File not found: \n{}'.format(img2)
    assert osp.exists(msk1), 'File not found: \n{}'.format(msk1)
    assert osp.exists(msk2), 'File not found: \n{}'.format(msk2)

    cmd = './{} {} {} -nt 0 -out {} '.format(dm_bin, img1, img2, out_file)
    begin = time.time()
    # call the deep matching module from shell
    status = call(cmd, shell=True)
    assert status == 0, \
        'Deep matching exited with code {:d}. The command is \n{}'.format(status, cmd)

    # Filter constraints to  belong in the same segment type and within radius 60px
    # Load mask1 and mask2 images
    msk1 = np.array(Image.open(msk1))
    msk2 = np.array(Image.open(msk2))
    lines = open(out_file).read().splitlines()
    cstrs = []
    # check the constraints
    for line in lines:
        x1, y1, x2, y2 = [int(l) for l in line.split(' ')[:4]]
        if sqrt((x2-x1)**2 + (y2-y1)**2) < 60 and msk1[y1, x1] == msk2[y2, x2]:
            cstrs.append('\t'.join(['{:d}']*4).format(x1, y1, x2, y2))
    # write back to file
    open(out_file, 'w').write('\n'.join([str(len(cstrs))] + cstrs))
    elapsed = time.time() - begin
    return elapsed

def matching(fd, rgb_root, msk_root, cst_root):

    # TODO have the file pattern input from argument
    reg = re.compile('(\d+)\.jp.?g', flags=re.IGNORECASE) # or put (?i)jp.g
    for root, dirs, _ in os.walk(rgb_root):
        # check if the folder contain files of the wanted pattern
        for d in dirs:
            files = [f for f in os.listdir(osp.join(root, d)) if reg.search(f) is not None]
            if len(files) == 0:
                continue
            # if there is, then get the file list and check the next frame
            for f in files:
                num = reg.search(f)
                if num is None:
                    continue
                n = '{:0'+str(len(num.group(1)))+'d}'
                nxt = int(num.group(1))+fd # next frame according to frame distance fd
                f2 = f.replace(num.group(1), n.format(nxt))
                if not osp.exists(osp.join(root, d, f2)):
                    continue
                if not osp.isdir(osp.join(cst_root, d)):
                    os.makedirs(osp.join(cst_root, d))
                run_matching(osp.join(root, d, f),
                            osp.join(root, d, f2),
                            osp.join(msk_root, d, osp.splitext(f)[0]+'.png'),
                            osp.join(msk_root, d, osp.splitext(f2)[0]+'.png'),
                            osp.join(cst_root, d, osp.splitext(f)[0]+'.txt'))

def main(flags):

    org_color_root = osp.join(input_root, orgcolor)
    org_mask_root = osp.join(input_root, orgmask)
    constraint_root = osp.join(output_root, constraints_dir)
    flow_root = osp.join(output_root, flow_dir)


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


    print 'Image matching',
    sys.stdout.flush()
    begin = time.time()
    matching(1, org_color_root , org_mask_root, constraint_root)
    print "\t[Done] | {:.2f} mins".format((time.time()-begin)/60)
    sys.stdout.flush()

    # TODO check if input images are jpg and convert to png
    print 'Converting original images',
    sys.stdout.flush()
    begin = time.time()
    convert_rgb(org_color_root, im1paths['rgb_root'])
    convert_mask(org_mask_root, im1paths['mask_root'])
    print "\t[Done] | {:.2f} mins".format((time.time()-begin)/60)
    sys.stdout.flush()

    print 'Image ARAP deformation',
    sys.stdout.flush()
    begin = time.time()
    arap_deform(im1paths['rgb_root'], im1paths['mask_root'],
                constraint_root, flow_root,
                im2paths['rgb_root'], im2paths['mask_root'])
    print "\t[Done] | {:.2f} mins".format((time.time()-begin)/60)
    print 'Adding static background',
    lines = bg_gen(bg_dir, im1paths, im2paths, flow_root)
    open(osp.join(output_root, 'all_files.list'), 'w').write('\n'.join(lines))



if __name__ == "__main__":
    # TODO check if rgb and mask images are in png format, if not convert them to png
    # TODO input if want to keep constraints and warped mask
    parser = argparse.ArgumentParser(description='Argument for ARAP flow generation')
    parser.add_argument('--rm-cnstr')
    parser.add_argument('--rm-wmask')
    parser.add_argument('--rm-tmp-cmd')
    parser.add_argument('--img-pattern')
    flags = parser.parse_args()
    main(flags)
