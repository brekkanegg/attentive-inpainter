import os, sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['places365', 'coco'])
parser.add_argument('--which_gpu', type=str, default='0')
parser.add_argument('--demo', type=str, choices=["0", "1", "2", "3", "4"])


parser.add_argument('--checkpointDir', type=str, default=None)
# parser.add_argument('--maskType', type=str,
#                     choices=['random', 'center', 'left', 'full', 'grid', 'lowres'],
#                     default='center')
# parser.add_argument('--centerScale', type=float, default=0.25)
# parser.add_argument('--use_attention', type=bool, default=False)
# parser.add_argument('--attention_grid', type=int, default=3)




args = parser.parse_args()


args.checkpointDir = os.path.join('checkpoint', args.dataset, 'train_64')
args.dataset = os.path.join('../data', args.dataset, 'demo', args.demo)

maskTypes = ['center', 'left', 'random']
centerScales = [0.15, 0.25, 0.35]
use_attentions = [True, False]
attention_grids = [3]

for mt in maskTypes:
    for cs in centerScales:
        for ua in use_attentions:
            for ag in attention_grids:
                os.system('python complete.py --which_gpu {} --checkpointDir {} --dataset {} --maskType {} '
                          '--centerScale {} --use_attention {} --attention_grid {}'.
                          format(args.which_gpu, args.checkpointDir, args.dataset, mt, cs, ua, ag))