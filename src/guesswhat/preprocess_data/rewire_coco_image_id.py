import re
import os
from glob import glob

import argparse

parser = argparse.ArgumentParser(description="Python make_cococaption_id_names.py {cococaption basedir}\n\n"
          "After unpacking MS COCO zip files, you should end up with a\n"
          "base directory with train2014, valid2014 and test2014 folder in it.\n"
          "Provide this base dir as the first argument to this script and it\n"
          "will create symbolic links in the 'raw' folder, where each\n"
          "link name corresponds to the original image's ID.\n"
          "This format is expected by the Guesswhat?! preprocessing.")

parser.add_argument("-image_dir", type=str, required=True, help="Input Image folder (WARNING absolute path)")
parser.add_argument("-image_subdir", type=list, default=["train2014", "val2014", "test2014"], help='Select the dataset subdir')
parser.add_argument("-data_out", type=str, required=True, help="Output symlink folder (WARNING absolute path)")

args = parser.parse_args()

assert args.image_dir.startswith(os.path.sep), "The path must be a root path: ".format(args.image_dir)
assert args.data_out.startswith(os.path.sep), "The path must be a root path: ".format(args.data_out)

for path in args.image_subdir:
    for name in glob(os.path.join(args.image_dir, path, "*")):

        # retrieve id images
        res = re.match(r'.*_0*(\d+\.\w+)$', name)
        if not res:
            continue

        # create symlink with id_picture
        os.symlink(name, os.path.join(args.data_out, res.group(1)))
