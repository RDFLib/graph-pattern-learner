# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import logging
import random
from collections import OrderedDict
from collections import defaultdict
from functools import partial
from operator import itemgetter

import config
from serialization import load_fusion_model
from serialization import save_fusion_model
from utils import exception_stack_catcher
from .basic import Fusion
from .basic import basic_fm
from .trained import classifier_fm
from .trained import classifier_fm_fast
from .trained import classifier_fm_slow
from .trained import regression_fm_fast
from .trained import regression_fm_slow
from .trained import regression_fm
from .trained import ranksvm_fm
from .vecs import gp_tcs_to_vecs
from .vecs import prep_training
from .vecs import vecs_labels_to_unique_vecs_ratio

logger = logging.getLogger(__name__)


# noinspection PyTypeChecker
all_fusion_methods = OrderedDict(
    (_fm.name, _fm) for _fm in
    basic_fm + classifier_fm + regression_fm + ranksvm_fm
)


def get_fusion_methods_from_str(fms_arg=None):
    if not fms_arg:
        fms_arg = 'default'
    # default = [
    #     'basic',
    #     'svm_linear', 'svm_rbf', 'gtb', 'neural_net', 'logistic_regression',
    #     'rank_svm',
    # ]
    default = ['all']

    fmsl = [s.strip() for s in fms_arg.split(',')]
    if 'default' in fmsl:
        # replace 'default' with its method names
        i = fmsl.index('default')
        fmsl[i:i+1] = default

    # replace with fusion methods, also expanding 'basic' and 'classifiers'
    fml = []
    for s in fmsl:
        if s == 'basic':
            fml.extend(basic_fm)
        elif s == 'classifiers':
            fml.extend(classifier_fm)
        elif s == 'classifiers_fast':
            fml.extend(classifier_fm_fast)
        elif s == 'classifiers_slow':
            fml.extend(classifier_fm_slow)
        elif s == 'regressors':
            fml.extend(regression_fm)
        elif s == 'regressors_fast':
            fml.extend(regression_fm_fast)
        elif s == 'regressors_slow':
            fml.extend(regression_fm_slow)
        elif s == 'all':
            fml.extend(all_fusion_methods.values())
        elif s.startswith('-'):
            # allows to remove individual fusion methods
            # "all,-svr_linear,-svr_rbf"
            s = s[1:]
            try:
                fm = all_fusion_methods[s]
            except KeyError:
                logger.error(
                    'unknown negative fusion method: %s\navailable: %s',
                    s,
                    all_fusion_methods.keys()
                )
                raise
            try:
                fml.remove(fm)
            except ValueError:
                logger.warning(
                    "%s not found in prior fusion methods %s, skipping", s, fml)
        else:
            try:
                fml.append(all_fusion_methods[s])
            except KeyError:
                logger.error(
                    'unknown fusion method: %s\navailable: %s',
                    s,
                    [
                        'basic',
                        'classifiers',
                        'classifiers_fast',
                        'classifiers_slow',
                        'regressors',
                        'regressors_fast',
                        'regressors_slow',
                    ] + all_fusion_methods.keys()
                )
                raise

    return fml


def train_fusion_models(gps, gtps, target_candidate_lists, fusion_methods=None):
    training_tuple = prep_training(gps, gtps, target_candidate_lists)
    for fm in get_fusion_methods_from_str(fusion_methods):
        fm.train(gps, gtps, target_candidate_lists, training_tuple)


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

    targets_vecs = gp_tcs_to_vecs(gps, target_candidate_lists)
    res = OrderedDict()
    for fm in get_fusion_methods_from_str(fusion_methods):
        try:
            res[fm.name] = fm.fuse(gps, target_candidate_lists, targets_vecs)
        except NotImplementedError:
            logger.warning(
                'seems %s is not implemented yet, but called', fm.name
            )
            res[fm.name] = []
    return res
