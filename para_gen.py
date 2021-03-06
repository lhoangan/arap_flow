import re, os, sys, time, numpy as np, random as rn, os.path as osp
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

parser = argparse.ArgumentParser(description='Arguments for ARAP flow generation')

#=============================================================================

def fit_bg(bg, im):
    imh, imw = im.shape[:2]
    bgh, bgw = bg.shape[:2]
    bgim = Image.fromarray(bg)

    hmax = max(bgh, imh)
    wmax = max(bgw, imw)
    r = rn.uniform(1, 2) * max(float(hmax)/bgh, float(wmax)/bgw)
    bgim = bgim.resize((int(bgw*r), int(bgh*r)), Image.ANTIALIAS)
    bg = np.array(bgim)
    # random crop the background to the image
    sy, sx = rn.randint(0, bg.shape[0] - imh), rn.randint(0, bg.shape[1] - imw)
    return bg[sy:(sy+imh), sx:(sx+imw), :]

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
        flow_path, rgb2_path, msk2_path = seg_paths[0].split(' ')[-3:]
        flow_im = np.dstack(sintel_io.flow_read(flow_path))
        rgb2_im = np.array(Image.open(rgb2_path))
        msk2_im = np.array(Image.open(msk2_path))

        if len(rgb2_im.shape) == 2:
            rgb2_im = rgb2_im[..., None]

        os.remove(flow_path)
        os.remove(rgb2_path)
        os.remove(msk2_path)

        for i in range(1, len(seg_paths)):
            flow_path, rgb2_path, msk2_path = seg_paths[i].split(' ')[-3:]
            flow_ = np.dstack(sintel_io.flow_read(flow_path))
            rgb2_ = np.array(Image.open(rgb2_path))
            msk2_ = np.array(Image.open(msk2_path))
            msk_ob = msk2_ != 0
            msk_bg = msk2_ == 0

            if len(rgb2_.shape) == 2:
                rgb2_ = rgb2_[..., None]

            flow_im = flow_im*msk_bg[..., None] + flow_ * msk_ob[..., None]
            rgb2_im = rgb2_im*msk_bg[..., None] + rgb2_ * msk_ob[..., None]
            msk2_im = msk2_im*msk_bg + msk2_ * msk_ob

            os.remove(flow_path)
            os.remove(rgb2_path)
            os.remove(msk2_path)

        # output
        sintel_io.flow_write(arap_path.split(' ')[-3], flow_im)
        Image.fromarray(rgb2_im).save(arap_path.split(' ')[-2])
        Image.fromarray(msk2_im.astype(np.uint8)).save(arap_path.split(' ')[-1])

    return [entry[0] for entry in arap_seg_paths]


def do_arap(paths, bgs, gpu, gpu_queue, arap_seg_paths):

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
    for path, bg in zip(paths, bgs):
        pt, mk = path.split(' ')[-2:]
        im = np.array(Image.open(pt))
        mk = np.array(Image.open(mk))
        im = add_bg(im, mk, bg)
        Image.fromarray(im).save(pt)

    gpu_queue.put(gpu)

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
    if flags.size is not None and im.size != flags.size:
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
        arap_path = make_arap_path(p)
        lmdb_paths.append(' '.join([arap_path.split(' ')[l] for l in [0, 4, 3]]))

        # preparing for output
        for k in p:
            if not osp.isdir(osp.dirname(p[k])):
                os.makedirs(osp.dirname(p[k]))

        im1, mk1, im2, mk2 = preprocess(p)

        if not has_mask(p['msk1_org'], p['msk2_org']):
            cleanup(p)
            continue

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
            continue

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
        bgim = fit_bg(bgim, im1)
        out1 = add_bg(im1, mk1, bgim)

        # output to file
        bgs.append(bgim)
        if not osp.isdir(osp.dirname(p['rgb1_gen'])):
            os.makedirs(osp.dirname(p['rgb1_gen']))
        Image.fromarray(out1).save(p['rgb1_gen'])

        # Convert mask
        if not osp.isdir(osp.dirname(p['msk1_gen'])):
            os.makedirs(osp.dirname(p['msk1_gen']))

        seg_paths = None
        if not flags.multseg:
            mask = np.zeros_like(mk1, dtype=np.uint8)
            mask[mk1==0] = ARAP_BG # TODO mask with each object segment separately
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


        #do_arap(arap_paths, bgs, 0, arap_seg_paths)
        #arap_paths = []
        #arap_seg_paths = []
        #bgs = []

        if not gpu_queue.empty():
            gpu = gpu_queue.get()
            proc = Process(target=do_arap, args=(arap_paths, bgs, gpu, gpu_queue, arap_seg_paths))
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
    main()
