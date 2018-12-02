
from joblib import Parallel, delayed
from subprocess import call
import multiprocessing
import os, os.path as osp, time

ARAP_DIR = 'Warp/deformation/'
DEEPMATCH_DIR = '/home/hale/TrimBot/projects/optflow/deepmatch/deepmatching-static'

def arap(path, progress):
    cmd = 'cd {} ; ./image_warping {:s}'.format(ARAP_DIR, path)

    begin = time.time()
    call(cmd, shell=True)
    print '[{:.2f}% ] {} | {:.3f}s'.format(progress*100, cmd, time.time() - begin)



# Prepare list of paths for warping command
rgb_root = '/home/hale/Datasets/MPI-Sintel-flow/training/'
msk_root = '/home/hale/TrimBot/projects/optflow/fgbg_inv/'
cst_root = '/home/hale/TrimBot/projects/optflow/fgbg_match/'
flo_root = '/home/hale/TrimBot/projects/optflow/fgbg_flo'
wrp_root = '/home/hale/TrimBot/projects/optflow/fgbg_msk'
wmk_root = '/home/hale/TrimBot/projects/optflow/fgbg_wrp'

paths = []
for p in ['clean', 'final']:
    for seq in ['ambush_6',  'bamboo_2',  'cave_4',  'market_6',  'temple_2']:
        for f in os.listdir(osp.join(cst_root, p, seq)):
            if '.txt' not in f:
                continue
            path = {}
            fn = f.replace('.txt', '.png')
            ff = f.replace('.txt', '.flo')
            path['rgbp'] = osp.join(rgb_root, p, seq, fn)
            path['mskp'] = osp.join(msk_root, seq, fn)
            path['cstp'] = osp.join(cst_root, p, seq, f)
            path['flop'] = osp.join(flo_root, p, seq, ff)
            path['wrpp'] = osp.join(wrp_root, p, seq, fn)
            path['wmkp'] = osp.join(wmk_root, p, seq, fn)

            assert osp.exists(path['rgbp']) and \
                osp.exists(path['mskp']) and osp.exists(path['cstp']), \
                'Files not found: \n'+ \
                '\n'.join([path['rgbp'], path['mskp'], path['cstp']])

            paths.append(path)

            if not osp.isdir(osp.dirname(path['flop'])):
                os.makedirs(osp.dirname(path['flop']))
            if not osp.isdir(osp.dirname(path['wrpp'])):
                os.makedirs(osp.dirname(path['wrpp']))
            if not osp.isdir(osp.dirname(path['wmkp'])):
                os.makedirs(osp.dirname(path['wmkp']))


num_cores = multiprocessing.cpu_count()
num_cores = 7
n = float(len(paths))
print n
if not osp.isdir('tmp'):
    os.makedirs('tmp')
files = []
for i in range(int(n/num_cores) + 1):
    cmds = []
    for j in range(i*num_cores, int(min((i+1)*num_cores, n))):
        path = paths[j]
        cmd = '{:s} {:s} {:s} {:s} {:s} {:s}'.format(
            osp.join(path['rgbp']),
            osp.join(path['mskp']),
            osp.join(path['cstp']),
            osp.join(path['flop']),
            osp.join(path['wrpp']),
            osp.join(path['wmkp']))
        cmds.append(cmd)
    open('tmp/{:d}.png'.format(i), 'w').write('\n'.join(cmds))
    files.append(osp.abspath('tmp/{:d}.png'.format(i)))

Parallel(n_jobs = num_cores)(delayed(arap)(p, i/n) for i, p in enumerate(files))
