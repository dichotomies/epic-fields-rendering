import argparse
from .meta import *
from .frames import *


parser = argparse.ArgumentParser()


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--frames_selection",
        default="3dv",
        choices=["3dv", "all", "dbg"],
        type=str,
    )

    parser.add_argument(
        "--colmap_dir",
        type=str,
        help="Root directory of COLMAP project directory. For example `data/P01/P01_01/colmap/P01_01-s01.00-n00003dv`. `sparse/0` is selected by default.",
    )

    parser.add_argument(
        "--scale", default=1, type=int, help="Downscaling factor for images."
    )

    parser.add_argument(
        "--dir_dst", default='data/custom', type=str, help="Destination directory for converted dataset."
    )

    parser.add_argument(
        "--split_nth", default=0, type=int, help="select every n-th frame as validation and every other n-th frame as test frame."
    )

    parser.add_argument("-f", default=None, help="Jupyter compatibility.")

    args = parser.parse_args()

    return args

def run(args):
    colmap_model_dir = os.path.join(args.colmap_dir, 'sparse/0')
    colmap = load_colmap(colmap_model_dir)
    meta = calc_meta(colmap, split_nth=args.split_nth)
    frames_dir_src = os.path.join(args.colmap_dir, 'frames')
    dataset_id = os.path.split(os.path.normpath(args.colmap_dir))[1]
    dataset_dir = os.path.join(args.dir_dst, dataset_id)
    frames_dir_dst = os.path.join(dataset_dir, 'frames')
    os.makedirs(frames_dir_dst)

    save_meta(dataset_dir, meta)
    save_frames(frames_dir_src, frames_dir_dst, meta)

if __name__ == '__main__':
    args = parse_args()
    run(args)
