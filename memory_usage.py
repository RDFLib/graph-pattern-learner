# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import gc
import random
from six.moves import StringIO

import objgraph

from logging_config import filename

_count = 0


def log_mem_usage(signum, frame, fname=None):
    global _count
    _count += 1
    gc.collect()
    if not fname:
        fname = filename + '_memory_%02d.log' % _count
    with open(fname, 'wb') as f:
        f.write('gc.garbage: %d\n\n' % len(gc.garbage))
        objgraph.show_most_common_types(limit=50, file=f)
        f.write('\n\n')
        buf = StringIO()
        objgraph.show_growth(limit=50, file=buf)
        buf = buf.getvalue()
        f.write(buf)
    if _count < 2:
        return
    for tn, l in enumerate(buf.splitlines()[:10]):
        l = l.strip()
        if not l:
            continue
        type_ = l.split()[0]
        objects = objgraph.by_type(type_)
        objects = random.sample(objects, min(50, len(objects)))
        objgraph.show_chain(
            objgraph.find_backref_chain(
                objects[0],
                objgraph.is_proper_module),
            filename=fname[:-4] + '_type_%02d_backref.png' % tn
        )
        objgraph.show_backrefs(
            objects,
            max_depth=5,
            extra_info=lambda x: hex(id(x)),
            filename=fname[:-4] + '_type_%02d_backrefs.png' % tn,
        )
