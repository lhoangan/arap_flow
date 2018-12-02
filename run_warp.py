
from joblib import Parallel, delayed
from subprocess import call
import multiprocessing
import os, os.path as osp, time

WARP_DIR = 'Warp/warping'

def warp(path, progress):
    cmd = './{}/image_warping {:s} {:s} {:s} {:s} {:s}'.format(
        WARP_DIR,
        osp.join(path['rgbp']),
        osp.join(path['mskp']),
        osp.join(path['flop']),
        osp.join(path['wrpp']),
        osp.join(path['wmkp']))

    begin = time.time()
    call(cmd, shell=True)
    print '[{:.2f}% ] {} | {:.3f}s'.format(progress*100, cmd, time.time() - begin)



# Prepare list of paths for warping command
rgb_root = '/media/hale/datastore/Datasets/DAVIS_480/image'
msk_root = '/media/hale/datastore/Datasets/DAVIS_480/segments'
flo_root = '/media/hale/datastore/Datasets/ARAP_flow/DAVIS-FN2/flow'
wrp_root = '/media/hale/datastore/Datasets/ARAP_flow/DAVIS-FN2/warped-rgb'
wmk_root = '/media/hale/datastore/Datasets/ARAP_flow/DAVIS-FN2/warped-masks'


paths = []
for fd in [1, 2, 3, 4, 5, 9, 13]:
    fd_dir = 'DAVIS_%d'%fd
    flo_path = osp.join(flo_root, fd_dir)
    wrp_path = osp.join(wrp_root, fd_dir)
    wmk_path = osp.join(wmk_root, fd_dir)
    for seq in os.listdir(flo_path):
        if not osp.isdir(osp.join(flo_path, seq)):
            continue
        for f in os.listdir(osp.join(flo_path, seq)):
            if '.flo' not in f:
                continue
            path = {}
            fn = f.replace('.flo', '.png')
            path['rgbp'] = osp.join(rgb_root, seq, fn)
            path['mskp'] = osp.join(msk_root, seq, fn)
            path['flop'] = osp.join(flo_path, seq, f)
            path['wrpp'] = osp.join(wrp_path, seq, fn)
            path['wmkp'] = osp.join(wmk_path, seq, fn)

            assert osp.exists(path['rgbp']) and \
                osp.exists(path['mskp']) and osp.exists(path['flop']), \
                'Files not found: \n'+ \
                '\n'.join([path['rgbp'], path['mskp'], path['flop']])

            paths.append(path)

            if not osp.isdir(osp.dirname(path['wrpp'])):
                os.makedirs(osp.dirname(path['wrpp']))
            if not osp.isdir(osp.dirname(path['wmkp'])):
                os.makedirs(osp.dirname(path['wmkp']))


num_cores = multiprocessing.cpu_count()
n = float(len(paths))
Parallel(n_jobs = num_cores)(delayed(warp)(p, i/n) for i, p in enumerate(paths))
