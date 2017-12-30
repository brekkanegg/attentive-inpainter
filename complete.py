# todo: fix to apply for other image size, or change image size to 64

#!/usr/bin/env python3
#
# Brandon Amos (http://bamos.github.io)
# License: MIT
# 2016-08-05

import argparse
import os
import tensorflow as tf
import glob
from time import time

from model import DCGAN

parser = argparse.ArgumentParser()
parser.add_argument('--approach', type=str,
                    choices=['adam', 'hmc'],
                    default='adam')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--hmcBeta', type=float, default=0.2)
parser.add_argument('--hmcEps', type=float, default=0.001)
parser.add_argument('--hmcL', type=int, default=100)
parser.add_argument('--hmcAnneal', type=float, default=1)
parser.add_argument('--nIter', type=int, default=1000) # fixme
parser.add_argument('--imgSize', type=int, default=64)
parser.add_argument('--lam', type=float, default=0.1)
parser.add_argument('--checkpointDir', '--checkpoint', type=str, default='checkpoint')
parser.add_argument('--outDir', type=str, default='completions')
parser.add_argument('--outInterval', type=int, default=20)
parser.add_argument('--maskType', type=str,
                    choices=['random', 'center', 'left', 'full', 'grid', 'lowres'],
                    default='center')
parser.add_argument('--centerScale', type=float, default=0.25)

parser.add_argument('--dataset', type=str)
parser.add_argument('--imgs', type=str, nargs='+')

parser.add_argument('--which_gpu', type=str, default='0')
parser.add_argument('--use_attention', type=str, default='False')
parser.add_argument('--attention_grid', type=int, default=3)


args = parser.parse_args()

assert(os.path.exists(args.checkpointDir))

os.environ['CUDA_VISIBLE_DEVICES'] = args.which_gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

args.imgs = glob.glob(os.path.join(args.dataset, '*.jpg'))
args.outDir = os.path.join(args.outDir, args.dataset.split('data/')[1],
                           args.maskType, str(args.centerScale), args.use_attention, str(args.attention_grid))

args.use_attention = True if args.use_attention == 'True' else False

with tf.Session(config=config) as sess:
    dcgan = DCGAN(sess, image_size=args.imgSize,
                  batch_size=min(64, len(args.imgs)),
                  checkpoint_dir=args.checkpointDir, lam=args.lam)
    t1 = time()
    dcgan.complete(args)
    t2 = time()
    print('completion time: ', t2-t1)