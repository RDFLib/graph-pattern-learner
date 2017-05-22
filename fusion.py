# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


def fuse_prediction_results(predict_query_results, fusion_methods=None):
    """Several naive prediction methods for targets given a source.

    The naive methods here are used to re-assemble all result lists returned by
    each of the GraphPatterns in gps. Some of them use the fitness information
    of the gps.
    The methods used are:
        - 'target_occs': a simple occurrence count of the target over all gps.
        - 'scores': sum of all gp scores for each returned target.
        - 'f_measures': sum of all gp f_measures for each returned target.
        - 'gp_precisions': sum of all gp precisions for each returned target.
        - 'precisions': sum of the actual precisions per gp in this prediction.
        - 'target_occs_precisions': same as above but scaled with precision
        - 'scores_precisions': same as above but scaled with precision
        - 'f_measures_precisions': same as above but scaled with precision
        - 'gp_precisions_precisions': same as above but scaled with precision

    :param predict_query_results: a list of [(target_candidates, gp)] as
        returned by predict_target_candidates().
    :param fusion_methods: None for all or a list of strings naming the fusion
        methods to return.
    :return: A dict like {method: ranked_res_list}, where ranked_res_list is a
        list result list produced by method of (predicted_target, score) pairs
        ordered decreasingly by score. For methods see above.
    """
    target_occs = Counter()
    scores = Counter()
    f_measures = Counter()
    gp_precisions = Counter()
    precisions = Counter()
    target_occs_precisions = Counter()
    scores_precisions = Counter()
    f_measures_precisions = Counter()
    gp_precisions_precisions = Counter()

    # TODO: add cut-off values for methods (will have different recalls then)

    for gp, res in predict_query_results:
        score = gp.fitness.values.score
        fm = gp.fitness.values.f_measure
        gp_precision = 1
        avg_reslens = gp.fitness.values.avg_reslens
        if avg_reslens > 0:
            gp_precision = 1 / avg_reslens
        n = len(res)
        precision = 1
        if n > 0:
            precision = 1 / n
        for t in res:
            target_occs[t] += 1
            scores[t] += score
            f_measures[t] += fm
            gp_precisions[t] += gp_precision
            precisions[t] += precision
            target_occs_precisions[t] += 1 * precision
            scores_precisions[t] += score * precision
            f_measures_precisions[t] += fm * precision
            gp_precisions_precisions[t] += gp_precision * precision
    res = OrderedDict([
        ('target_occs', target_occs.most_common()),
        ('scores', scores.most_common()),
        ('f_measures', f_measures.most_common()),
        ('gp_precisions', gp_precisions.most_common()),
        ('precisions', precisions.most_common()),
        ('target_occs_precisions', target_occs_precisions.most_common()),
        ('scores_precisions', scores_precisions.most_common()),
        ('f_measures_precisions', f_measures_precisions.most_common()),
        ('gp_precisions_precisions', gp_precisions_precisions.most_common()),
    ])
    if fusion_methods:
        # TODO: could improve by not actually calculating them
        for k in res.keys():
            if k not in fusion_methods:
                del res[k]
    return res
