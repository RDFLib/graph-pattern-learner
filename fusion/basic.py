# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from collections import Counter
from operator import mul

logger = logging.getLogger(__name__)


class Fusion(object):
    name = 'base_fusion'

    def train(self, gps, gtps, target_candidate_lists, training_tuple=None):
        pass

    def save(self, filename=None):
        pass

    def load(self, filename=None):
        pass

    def fuse(self, gps, target_candidate_lists, targets_vecs=None):
        raise NotImplementedError


class BasicWeightedFusion(Fusion):
    """Base for several naive weighted fusion methods.

    The naive methods here are used to re-assemble all result lists returned by
    each of the GraphPatterns in gps. Some of them use the fitness information
    of the gps, some the actual lengths, some both.
    """
    def __init__(
            self, name, desc,
            getter_gp=lambda gp: 1,
            getter_tcs=lambda gp: 1,
            combine_getters=mul,
    ):
        super(BasicWeightedFusion, self).__init__()
        self.name = name
        self.desc = desc
        self.getter_gp = getter_gp
        self.getter_tcs = getter_tcs
        self.combine = combine_getters

    def fuse(self, gps, target_candidate_lists, targets_vecs=None):
        c = Counter()
        for gp, tcs in zip(gps, target_candidate_lists):
            gpg = self.getter_gp(gp)
            for t in tcs:
                c[t] += self.combine(gpg, self.getter_tcs(tcs))
        return c.most_common()


def _score_getter(gp):
    return gp.fitness.values.score


def _f_measure_getter(gp):
    return gp.fitness.values.f_measure


def _gp_precisions_getter(gp):
    if gp.fitness.values.avg_reslens > 0:
        return 1 / gp.fitness.values.avg_reslens
    else:
        return 1


def _precisions_getter(tcs):
    if len(tcs) > 0:
        return 1 / len(tcs)
    else:
        return 1


basic_fm = [
    BasicWeightedFusion(
        'target_occs',
        'a simple occurrence count of the target over all gps.',
    ),
    BasicWeightedFusion(
        'scores',
        'sum of all gp scores for each returned target.',
        getter_gp=_score_getter,
    ),
    BasicWeightedFusion(
        'f_measures',
        'sum of all gp f_measures for each returned target.',
        getter_gp=_f_measure_getter,
    ),
    BasicWeightedFusion(
        'gp_precisions',
        'sum of all gp precisions for each returned target.',
        getter_gp=_gp_precisions_getter,
    ),
    BasicWeightedFusion(
        'precisions',
        'sum of the actual precisions per gp in this prediction.',
        getter_tcs=_precisions_getter,
    ),
    BasicWeightedFusion(
        'scores_precisions',
        'same as above but scaled with precision',
        getter_gp=_score_getter, getter_tcs=_precisions_getter,
    ),
    BasicWeightedFusion(
        'f_measures_precisions',
        'same as above but scaled with precision',
        getter_gp=_f_measure_getter, getter_tcs=_precisions_getter,
    ),
    BasicWeightedFusion(
        'gp_precisions_precisions',
        'same as above but scaled with precision',
        getter_gp=_gp_precisions_getter, getter_tcs=_precisions_getter,
    ),
]
