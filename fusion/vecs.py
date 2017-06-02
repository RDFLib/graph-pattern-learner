# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)


def prep_training(
        _, gtps, target_candidate_lists,
        print_vecs=False,
        warn_about_multiclass_vecs=False,
):
    """Used to convert training's target candidates for all gtps into vectors.

    Mainly out-sourced for efficiency reasons, as static for all trained fusion
    methods.

    :returns 5-tuple:
        vecs (numpy array),
        labels (numpy array boolean),
        a list of one target candidate URIRef per vector,
        a list of one gtp (the input) per vector,
        groups (numpy array of gtp ids (integers) for GroupKFold)
    """
    assert len(gtps) == len(target_candidate_lists)
    logger.info('transforming all candidate lists to vectors')
    vecs = []
    vtcs = []
    vgtps = []
    vgtp_idxs = []
    labels = []
    for (gtp_idx, gtp), gp_tcs in zip(enumerate(gtps), target_candidate_lists):
        source, target = gtp
        targets = set(tc for tcs in gp_tcs for tc in tcs)
        for t in targets:
            vec = tuple(t in tcs for tcs in gp_tcs)
            label = t == target
            vecs.append(vec)
            vtcs.append(t)
            vgtps.append(gtp)
            vgtp_idxs.append(gtp_idx)
            labels.append(label)

    if print_vecs:
        for v, l in zip(vecs, labels):
            print('%s: %s' % (l, [1 if x else 0 for x in v]))

    if warn_about_multiclass_vecs:
        # warn about vectors occurring several times with different labels
        c = Counter(vecs)
        pos = Counter(vec for vec, l in zip(vecs, labels) if l)
        logger.info('unique vectors: %d, (total: %d)', len(c), len(vecs))
        for v, occ in c.most_common():
            if occ == 1:
                break
            if 0 < pos[v] < occ:
                logger.warning(
                    'same vector shall be classified differently:'
                    'pos: %d, neg: %d\n%s',
                    pos[v], occ - pos[v], [1 if x else 0 for x in v]
                )
    a = np.array
    return a(vecs, dtype='f8'), a(labels), vtcs, vgtps, a(vgtp_idxs)


def vecs_labels_to_unique_vecs_ratio(vecs, labels):
    """Groups identical training vectors and merges their labels into ratios.

    Called in training of classifiers (see config.FUSION_CMERGE_VECS) and in
    training of regressors.
    """
    vecs = [tuple(v) for v in vecs]
    c = Counter(vecs)
    p = Counter(vec for vec, l in zip(vecs, labels) if l)
    ratio = [p[vec]/t for vec, t in c.items()]
    return np.array(c.keys(), dtype='f8'), np.array(ratio)


def gp_tcs_to_vecs(_, gp_tcs):
    """Converts a list of target candidates per graph pattern into vectors.

    Called at fusion time.
    """
    targets = sorted(set(tc for tcs in gp_tcs for tc in tcs))
    vecs = []
    for t in targets:
        vec = tuple(t in tcs for tcs in gp_tcs)
        vecs.append(vec)
    return targets, np.array(vecs, dtype='f8')
