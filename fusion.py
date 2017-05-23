# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
from collections import OrderedDict
from operator import mul
import logging

logger = logging.getLogger(__name__)


class FusionModel(object):
    name = 'base'

    def train(self, gps, target_candidate_lists, stps):
        pass

    def fuse(self, gps, target_candidate_lists):
        pass


class BasicWeightedFusion(FusionModel):
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

    def fuse(self, gps, target_candidate_lists):
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


_bwf = BasicWeightedFusion
basic_fusion_methods = [
    _bwf('target_occs',
         'a simple occurrence count of the target over all gps.'),
    _bwf('scores',
         'sum of all gp scores for each returned target.',
         getter_gp=_score_getter),
    _bwf('f_measures',
         'sum of all gp f_measures for each returned target.',
         getter_gp=_f_measure_getter),
    _bwf('gp_precisions',
         'sum of all gp precisions for each returned target.',
         getter_gp=_gp_precisions_getter),
    _bwf('precisions',
         'sum of the actual precisions per gp in this prediction.',
         getter_tcs=_precisions_getter),
    _bwf('scores_precisions',
         'same as above but scaled with precision',
         getter_gp=_score_getter, getter_tcs=_precisions_getter),
    _bwf('f_measures_precisions',
         'same as above but scaled with precision',
         getter_gp=_f_measure_getter, getter_tcs=_precisions_getter),
    _bwf('gp_precisions_precisions',
         'same as above but scaled with precision',
         getter_gp=_gp_precisions_getter, getter_tcs=_precisions_getter),
]


_fusion_methods = OrderedDict([
    (bfm.name, bfm) for bfm in basic_fusion_methods
])


def fuse_prediction_results(gps, target_candidate_lists, fusion_methods=None):
    """

    :param gps: list of graph patterns.
    :param target_candidate_lists: a list of target_candidate lists as
        returned by predict_target_candidates() in same order as gps.
    :param fusion_methods: None for all or a list of strings naming the fusion
        methods to return.
    :return: A dict like {method: ranked_res_list}, where ranked_res_list is a
        list result list produced by method of (predicted_target, score) pairs
        ordered decreasingly by score. For methods see above.
    """
    assert len(gps) == len(target_candidate_lists)

    if fusion_methods is None:
        fusion_methods = _fusion_methods.values()
    else:
        assert fusion_methods and isinstance(fusion_methods[0], basestring)
        fusion_methods = [_fusion_methods[fmn] for fmn in fusion_methods]

    res = OrderedDict([
        (fm.name, fm.fuse(gps, target_candidate_lists))
        for fm in fusion_methods
    ])
    return res
