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
background_dir = '..'

orgcolor = 'orgRGB'
orgmask = 'orgMasks'
color_dir = 'inpRGB'
mask_dir = 'inpMasks'
constraints_dir = 'tmpCnstr'

tmp_color_dir = '..' # convert from JPG to PNG if needed
tmp_mask_dir = '..'

flow_dir = 'Flow'
wrgb_dir = 'wRGB'
wMask_dir = 'wMasks'

frame_distance = 1

# if constraints ava


sys.path.insert(0, '/home/hale/Datasets/MPI-Sintel-seg/sdk/python')
import sintel_io
sys.path.insert(0, '/home/hale/TrimBot/projects/myutils/')
import flow

# experiment setups
exp_num = '002-trial-10Lseg-seg.2'
#== ==========================================================================

def add_bg():
    pass


def run_arap(path, progress):
    cmd = '{} {:s}'.format(arap_bin, path)

    begin = time.time()
    status = call(cmd, shell=True)
    assert status == 0, \
        'ARAP exited with code {:d}. The command was \n{}'.format(status, cmd)
    print '[{:.2f}% ] | Elapsed {:.3f}s'.format(progress*100, time.time() - begin)

# TODO strip all the input path trailing slash
def prepare_arap(rgb_root, msk_root, cst_root, flo_root, wco_root, wmk_root):

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

    num_cores = int(multiprocessing.cpu_count() / 2)
    n = float(len(paths.keys()))
    #Parallel(n_jobs=num_cores)(delayed(run_arap)(p, i/n) for i, p in enumerate(files))
    run_arap(files[0], 100)

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
            mask[im==0] = 255 # TODO mask with each object segment separately
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

def prepare_matching(fd, rgb_root, msk_root, cst_root):

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
    prepare_matching(1, osp.join(input_root, orgcolor), osp.join(input_root, orgmask),
            osp.join(output_root, constraints_dir))
    # TODO check if input images are jpg and convert to png
    convert_rgb(osp.join(input_root, orgcolor), osp.join(input_root, color_dir))
    convert_mask(osp.join(input_root, orgmask), osp.join(input_root, mask_dir))
    prepare_arap(osp.join(input_root, color_dir),
            osp.join(input_root, mask_dir),
            osp.join(output_root, constraints_dir),
            osp.join(output_root, flow_dir),
            osp.join(output_root, wrgb_dir),
            osp.join(output_root, wMask_dir))


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
