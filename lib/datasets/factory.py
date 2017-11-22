# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import numpy as np

from .pascal_voc import pascal_voc
import os


# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012', '0712']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        print(name)
        __sets[name] = (lambda split=split, year=year:
                pascal_voc(split, year))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
    #if not __sets.has_key(name):
        print (list_imdbs())
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
