import re
import os
from glob import glob
import sys


if len(sys.argv) < 2:
    print("Usage: python make_cococaption_id_names.py {cococaption basedir}\n\n"
          "After unpacking cococaption zip files, you should end up with a\n"
          "cococaption base directory with train, valid and test folder in it.\n"
          "Provide this base dir as the first argument to this script and it\n"
          "will create symbolic links in a new 'plain' subfolder, where each\n"
          "link name corresponds to the original image's ID.\n"
          "This format is expected by the guesswhat preprocessing.")
    sys.exit(0)

basepath = sys.argv[1]

try:
    os.makedirs(os.path.join(basepath, "plain"))
except:
    pass

for path in "train2014 val2014 test2014".split():
    for name in glob(os.path.join(basepath, path, "*")):
        res = re.match(r'.*_0*(\d+\.\w+)$', name)
        if not res:
            continue
        os.symlink(name, os.path.join(basepath, "plain", res.group(1)))
