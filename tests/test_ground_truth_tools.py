# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from ground_truth_tools import get_semantic_associations
from ground_truth_tools import split_training_test_set
from ground_truth_tools import k_fold_cross_validation

logger = logging.getLogger(__name__)

associations = get_semantic_associations()


def test_split_train_test_set():
    vr = split_training_test_set(associations)
    train, test = vr
    logger.info("just random: train: %d, test: %d", len(train), len(test))
    vtnd = split_training_test_set(associations, variant='target_node_disjoint')
    train, test = vtnd
    logger.info("target node disjoint: train: %d, test: %d",
                len(train), len(test))
    vnd = split_training_test_set(associations, variant='node_disjoint')
    train, test = vnd
    logger.info("node disjoint: train: %d, test: %d", len(train), len(test))

    assert vr[0] == vtnd[0] == vnd[0], \
        "train set shouldn't be influenced by different splitting variant"
    assert set(vr[1]) > set(vtnd[1]) > set(vnd[1]), \
        "test set expected to shrink for more restrictive splitting variants"


def test_k_fold_cross_validation():

    l = associations
    k = 10

    vals = []
    sl = set(l)
    for t, v in k_fold_cross_validation(l, k, seed=None):
        st = set(t)
        sv = set(v)
        assert st | sv == sl, "train + validation != all"
        assert len(st & sv) == 0, "train and validation set overlap"
        vals += v
    assert set(vals) == sl, "all validation sets combined should be everything"




