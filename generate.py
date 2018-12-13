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

orgcolor = 'oRGB'
orgmask = 'oMasks'
color_dir = 'inRGB'
mask_dir = 'inMasks'
constraints_dir = 'inCnstr'

tmp_color_dir = '..' # convert from JPG to PNG if needed
tmp_mask_dir = '..'

flo_dir = '..'
wRGB_dir = '..'
wMask_dir = '..'

frame_distance = 1

# if constraints ava


sys.path.insert(0, '/home/hale/Datasets/MPI-Sintel-seg/sdk/python')
import sintel_io
sys.path.insert(0, '/home/hale/TrimBot/projects/myutils/')
import flow

# experiment setups
exp_num = '002-trial-10Lseg-seg.2'
#== ==========================================================================


def run_arap(path, progress):
    cmd = 'cd {} ; ./image_warping {:s}'.format(arap_bin, path)

    begin = time.time()
    status = call(cmd) #, shell=True)
    assert status == 0, \
        'ARAP exited with code {:d}. The command was \n{}'.format(status, cmd)
    print '[{:.2f}% ] | Elapsed {:.3f}s'.format(progress*100, time.time() - begin)

# TODO strip all the input path trailing slash
def prepare_arap(rgb_root, msk_root, cst_root, flo_root, wco_root, wmk_root):

    paths = dict()
    for root, dirs, files in os.walk(rgb_root):
        for f in files:
            if '.png' not in f:
                continue
            p = root.replace(rgb_root, '')
            s = os.dirname(p)
            paths[s] = paths.get(s, [])
            path = {}
            ft = f.replace('.png','.txt') # filename for text
            ff = f.replace('.png','.flo') # filename for flow
            path['rgb'] = osp.join(rgb_root, p, f)
            path['msk'] = osp.join(msk_root, p, f)
            path['cst'] = osp.join(cst_root, p, ft)
            path['flo'] = osp.join(flo_root, p, ff)
            path['wco'] = osp.join(wco_root, p, f)
            path['wmk'] = osp.join(wmk_root, p, f)

            assert osp.exists(path['rgb']), 'File not found:\n{}'.format(path['rgb'])
            assert osp.exists(path['msk']), 'File not found:\n{}'.format(path['msk'])
            assert osp.exists(path['cst']), 'File not found:\n{}'.format(path['cst'])

            if not osp.isdir(osp.dirname(path['flo'])):
                os.makedirs(osp.dirname(path['flo']))
            if not osp.isdir(osp.dirname(path['wco'])):
                os.makedirs(osp.dirname(path['wco']))
            if not osp.isdir(osp.dirname(path['wmk'])):
                os.makedirs(osp.dirname(path['wmk']))

            paths[s].append(path)

    # create temporary list file to input to ARAP
    if not osp.isdir('tmp'):
        os.makedirs('tmp')
    files = []
    # create temporary list files
    for key in paths:
        params_all = []
        path = paths[key]
        params_one = '{} {} {} {} {} {}'.format(osp.join(path['rgb']),
                                                osp.join(path['msk']),
                                                osp.join(path['cst']),
                                                osp.join(path['flo']),
                                                osp.join(path['wco']),
                                                osp.join(path['wmk']))
        params_all.append(params_one)
        # TODO use try and finally to clean the temporary files
        fn = key.replace('/', '_')
        open('tmp/{}.png'.format(fn), 'w').write('\n'.join(params_all))
        files.append(osp.abspath('tmp/{}.png'.format(fn)))

    num_cores = int(multiprocessing.cpu_count() / 2)
    n = float(len(paths.keys()))
    Parallel(n_jobs=num_cores)(delayed(run_arap)(p, i/n) for i, p in enumerate(files))

def convert_rgb(jpg_root, png_root):
    for root, _, files in os.walk(jpg_root):
        for f in files:
            if '.JPG' not in f.upper() or '.JPEG' not in f.upper():
                continue
            im = Image.open(osp.join(root, f))
            outdir = root.replace(jpg_root, png_root)
            if not osp.isdir(outdir):
                os.makedirs(outdir)
            im.save(osp.join(outdir, osp.splitext(f)+'.png'))

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
            Image.fromarray(mask).save(osp.join(out_root, osp.splitext(f)+'.png'))

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
    open(out_file, 'w').write('\n'.join(cstrs))
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
def add_bg():
    pass

def main():
    prepare_matching(1, osp.join(input_root, orgcolor), osp.join(input_root, orgmask),
            osp.join(output_root, constraints_dir))


if __name__ == "__main__":
    # TODO check if rgb and mask images are in png format, if not convert them to png
    main()

'''
#def make_cmd(path, fname,fseg,fmask1,fcons,fwarp,flow_dir, cmd='image_warping'):
def make_cmd(rgb_path, seg_path, mask_path, cstr_path,
        wrgb_path, wfield_path, wmask_path, path='.', cmd='image_warping'):
    args = '{:s} {:s} {:s} {:s} {:s} {:s} {:s}'.format(
        rgb_path,
        seg_path,
        mask_path,
        cstr_path,
        wrgb_path,
        wfield_path,
        wmask_path)

    cmd = osp.join(path, cmd) + ' ' + args
    return cmd

# get a random image (different from the one in maskout_dir) as background
def get_sintel_BG(img_root, maskout_dir):
    cands = []
    for root, dirs, files in os.walk(img_root):
        # remove all similar folders that share the same name stem: like alley_1, 2, 3,...
        if maskout_dir.split('_')[0] in root:
            continue
        # get list of all images
        for f in files:
            if '.png' not in f:
                continue
            cands.append(osp.join(root, f))
    # pick a random image and return
    return np.array(Image.open(cands[rn.randint(0, len(cands)-1)]))

# post processing
def post_process(wrgb_path, wfield_path, wmask_path, bg):

    # convert warp field to flow field
    # load warped field
    wfield = np.dstack(sintel_io.flow_read(wfield_path))
    w, h, _ = wfield.shape
    X, Y = np.meshgrid(np.arange(0, h), np.arange(0, w))
    flo = wfield - np.dstack((X, Y))
    flow_im = flow.to_color(flo)

    # output to files
    sintel_io.flow_write(wfield_path, flo)
    Image.fromarray(flow_im).save(wfield_path.replace('.flo', '.png'))

    # add background to warped image
    # load warped image
    wrgb = np.array(Image.open(wrgb_path))
    if len(wrgb.shape) == 2: # corrupted file
        print 'Found corrupted warped file. Removed generated file'
        os.remove(wrgb_path)
        os.remove(wfield_path)
        os.remove(wmask_path)
        return
    wmask = np.array(Image.open(wmask_path))
    wmask = (wmask > 0)[..., None].astype(np.uint8)

    im = wrgb * wmask + bg * (1 - wmask)
    Image.fromarray(im).save(wrgb_path)

def save_masked_org(rgb_path, mask_path, bg, rgb_out):
    rgb = np.array(Image.open(rgb_path))
    mask = np.array(Image.open(mask_path))

    mask = (mask == 0)[..., None].astype(np.uint8)

    im = rgb * mask + bg * (1 - mask)
    Image.fromarray(im).save(rgb_out)

# output masks for chosen segments
def make_mask(seg_im, s, out_path):

    mask = (seg_im != s).astype(np.uint8)*255
    out_dir = osp.dirname(out_path)
    if not osp.isdir(out_dir):
        os.makedirs(out_dir)
    Image.fromarray(mask).save(out_path)

# output constraints for chosen segments
def make_constraints(seg_im, s, matchings, out_path ):

    cstrs = []
    for match in matchings.astype(np.int).tolist():
        if seg_im[match[1], match[0]] == s:
            cstrs.append('{:d} {:d} {:d} {:d}'.format(*match[:4]))

    # randomly sample 20% of the matchings of THIS segment
    n = len(cstrs)
    if n > 0:
        idx = np.random.choice(np.arange(0, n), int(n*.2), replace=False)
        cstrs = [cstrs[i] for i in idx]

    # output
    out_dir = osp.dirname(out_path)
    if not osp.isdir(out_dir):
        os.makedirs(out_dir)
    with open(out_path, 'w') as fw:
        fw.write(str(len(cstrs)) + '\n')
        fw.write('\n'.join(cstrs))

def opt_warp(rgb_path, seg_path, mask_path, cstr_path,
        wrgb_path, wfield_path, wmask_path):

    # create output directories
    if not osp.isdir(osp.dirname(wrgb_path)):
        os.makedirs(osp.dirname(wrgb_path))
    if not osp.isdir(osp.dirname(wfield_path)):
        os.makedirs(osp.dirname(wfield_path))
    if not osp.isdir(osp.dirname(wmask_path)):
        os.makedirs(osp.dirname(wmask_path))

    # make command
    begin = time.time()
    cmd = make_cmd(rgb_path, seg_path, mask_path, cstr_path,
            wrgb_path, wfield_path, wmask_path)

    # call command to deform the segments
    status = call(cmd, shell=True)
    end = time.time()
    print '\n'*3, 'Run command: \n', cmd
    print '\nFinished opt warping. Elaspsed time: {:.3f} seconds'.format(end - begin), ' | Status: ', status

    return status

sintel_root = '/home/hale/Datasets/MPI-Sintel-flow/training/'
match_root = '/home/hale/Datasets/MPI-Sintel-flow/training/matching'

fdir = 'flow' # flow dir into sintel_root: sintel_root/fdir has alley_1...
passes = ['clean', 'final'] # seg/matching dirs into sintel_root or matching_root
seg_root = '/home/hale/Datasets/MPI-Sintel-seg/training/segmentation/'


# output directories of the script: input / output dirs for opt
opt_root = '/home/hale/Datasets/optflow/'
inp_cst = 'inp_cstr' # input dir of opt
inp_msk = 'inp_mask' # input dir of opt
out_rgb = 'out_rgb'  # output dir of opt
out_flo = 'out_flo'
out_msk = 'out_msk'

# 271 frames, 29881 segments (min=17 per im, max=775 per im, mean=294 per img)
val_set = ['ambush_2', 'bamboo_1', 'bandage_1', 'market_2', 'temple_2'] #'alley_1', 
out_root = osp.join(opt_root, exp_num)

begin = time.time()
counter = 0
for v in val_set:
    for p in passes:
        for root, dirs, files in os.walk(osp.join(match_root, p, v)):
            for ffull in files:
                if '.txt' not in ffull:
                    continue
                f = ffull.replace('.txt', '') # file stem, without extension

                counter += 1
                print 'Now process file #', counter, ': ', osp.join(root, f)

                # get matching result
                matchings = np.loadtxt(osp.join(root, f + '.txt'))

                # get segmentation image
                seg_im = sintel_io.segmentation_read(osp.join(
                    seg_root, v, f + '.png'))

                # list of segment indicess in this image
                segs = np.unique(seg_im)

                # sort the list of segments according to sizes
                _, srt_segs = zip(*sorted(zip([(seg_im==s).sum() for s in segs], segs)))

                # get 10 largest segments
                for s in srt_segs[-10:]:

                    sn = '_{:04d}'.format(s)

                    rgb_path = osp.join(sintel_root, p, v, f + '.png')
                    seg_path = osp.join(seg_root, v, f + '.png')
                    mask_path = osp.join(out_root, inp_msk, v, f + sn + '.png')
                    cstr_path = osp.join(out_root, inp_cst, p, v, f + sn + '.txt')
                    wrgb_path = osp.join(out_root, out_rgb, p, v, f + sn + '_1.png')
                    wfield_path = osp.join(out_root, out_flo, p, v, f + sn + '.flo')
                    wmask_path = osp.join(out_root, out_msk, p, v, f + sn + '.png')

                    # make mask for chosen segments
                    make_mask(seg_im, s, mask_path)

                    # make constraints for chosen images
                    make_constraints(seg_im, s, matchings, cstr_path)

                    # make command
                    status = opt_warp(
                        rgb_path, # rgb
                        seg_path, # seg
                        mask_path, # mask
                        cstr_path, # constraints
                        wrgb_path, # out warped rgb
                        wfield_path, # output warping field
                        wmask_path, # output warping field
                        )

                    if status != 0:
                        continue

                    # post processing
                    bg = get_sintel_BG(osp.join(sintel_root, p), v)
                    post_process(wrgb_path, wfield_path, wmask_path, bg)
                    # output original rgb image
                    save_masked_org(rgb_path, mask_path, bg, wrgb_path.replace('_1.png', '_0.png'))

end = time.time()
print '\n'*4, '='*10, '\n', 'Finished all in: {:.2f}mins'.format((end-begin)/60)
'''
