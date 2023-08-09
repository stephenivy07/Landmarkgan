import sys
sys.path.append('..')
from py_utils.vid_utils import proc_vid as pv
import numpy as np
import argparse


def main(args):
    imgs1, frame_num1, fps1, width1, height1 = pv.parse_vid(args.data_dir1)
    imgs2, frame_num2, fps2, width2, height2 = pv.parse_vid(args.data_dir2)

    num = np.minimum(frame_num1, frame_num2)
    vis_list = []
    for i in range(num):
        vis_im = np.concatenate([imgs1[i], imgs2[i]], axis=1)
        vis_list.append(vis_im)
    pv.gen_vid(video_path=args.output_path, imgs=vis_list, fps=fps1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generating video from images')
    parser.add_argument('--data_dir1', type=str)
    parser.add_argument('--data_dir2', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()
    main(args)


