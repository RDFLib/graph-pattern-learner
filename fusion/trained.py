# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import logging
import random
import warnings
from collections import OrderedDict
from collections import defaultdict
from datetime import datetime
from datetime import timedelta
from functools import partial
from operator import itemgetter

import numpy as np
from scoop.futures import map as parallel_map
from sklearn import clone
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lars
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import ParameterGrid
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

import config
from fusion.ranksvm import RankSVM
from serialization import load_fusion_model
from serialization import save_fusion_model
from utils import exception_stack_catcher

from .basic import Fusion
from .vecs import gp_tcs_to_vecs
from .vecs import prep_training
from .vecs import vecs_labels_to_unique_vecs_ratio

logger = logging.getLogger(__name__)


@exception_stack_catcher
def crossval_fm_score_single_split(
        split_tt_idxs,
        fm,
        clf,
        params,
        params_number,
        n_params,
        n_splits,
        vecs,
        labels,
        groups,
):
    split, (train_idxs, test_idxs) = split_tt_idxs
    # clf = clone(clf)
    logger.debug(
        '%s: param %d/%d: CV split %d/%d fitting...',
        fm.name, params_number, n_params, split, n_splits,
    )
    timer_start = datetime.utcnow()
    limit_reached = False
    # noinspection PyBroadException
    try:
        with warnings.catch_warnings(record=True) as w:
            clf.fit(vecs[train_idxs], labels[train_idxs])
            for warning in w:
                if issubclass(warning.category, ConvergenceWarning):
                    limit_reached = True
        res = fm.scores(
                vecs[test_idxs], labels[test_idxs], groups[test_idxs],
                clf=clf
        )
    except Exception:
        logger.warning(
            '%s: param %d/%d: CV split %d/%d: caused an exception\nparams: %s',
            fm.name, params_number, n_params, split, n_splits, params,
            exc_info=1
        )
        res = [-1]
    timer_diff = datetime.utcnow() - timer_start
    _lvl = logging.DEBUG
    reasons = []
    if timer_diff > timedelta(minutes=5):
        _lvl = logging.INFO
        reasons = ['took long']
    if timer_diff > timedelta(minutes=15):
        _lvl = logging.WARNING
        reasons = ['took long']
    if limit_reached:
        _lvl = logging.WARNING
        reasons.append('reached training iteration limit before convergence')
    r = ' and '.join(reasons)
    logger.log(
        _lvl,
        '%s: param %d/%d: CV split %d/%d %s: %s\nparams: %s',
        fm.name, params_number, n_params, split, n_splits, r, timer_diff, params
    )
    return timer_diff, res


@exception_stack_catcher
def crossval_fm_scores(numbered_params, fm, vecs, labels, groups, n_params):
    """Calculates the mean avg precision score for a given classifier via CV.

    Used at training time to find the MAP for the given classifier and params
    via cross validation.
    """
    assert isinstance(fm, FusionModel)
    params_number, params = numbered_params
    logger.debug(
        '%s: param %d/%d, starting CV for:\nparams: %s',
        fm.name, params_number, n_params, params)
    clf = clone(fm.clf)
    clf.set_params(**params)
    n_splits = config.FUSION_PARAM_TUNING_CV_KFOLD
    splits = GroupKFold(n_splits).split(vecs, labels, groups=groups)

    map_ = parallel_map if fm.parallelize_cv else map
    single_split_func = partial(
        crossval_fm_score_single_split,
        fm=fm,
        clf=clf,
        params=params,
        params_number=params_number,
        n_params=n_params,
        n_splits=n_splits,
        vecs=vecs,
        labels=labels,
        groups=groups,
    )
    cv_results = map_(single_split_func, enumerate(splits, 1))

    training_times, all_avg_precs = zip(*cv_results)
    all_avg_precs = list(itertools.chain.from_iterable(all_avg_precs))
    t = sum(training_times, timedelta())

    # if params_number % 10 == 0:
    #     logger.info('%s: completed crossval of param %d', name, params_number)
    mean_avg_prec = np.mean(all_avg_precs)
    std_avg_prec = np.std(all_avg_precs)
    logger.info(
        '%s: param %d/%d:\n'
        'params: %s\n'
        'CV results: score (MAP): %.3f (std: %.3f), training time: %s',
        fm.name, params_number, n_params, params, mean_avg_prec, std_avg_prec, t
    )
    return mean_avg_prec, std_avg_prec


def weight_multiplier(X, fm, weight):
    gps = fm.gps
    weights = np.array([getattr(gp.fitness.values, weight) for gp in gps])
    return X * weights


def weight_multiplier_gpprec(X, fm):
    gps = fm.gps
    avg_rl = lambda gp: gp.fitness.values.avg_reslens
    avg_res_lens = np.array([
        1 / avg_rl(gp) if avg_rl(gp) > 0 else 1
        for gp in gps])
    return X * avg_res_lens


class FusionModel(Fusion):
    name = 'FusionModel'
    train_on_ratios = False  # e.g., False for classifiers, True for regressors

    def __init__(self, name, clf, param_grid=None, parallelize_cv=True):
        self.name = name
        self.gps = None
        self.parallelize_cv = parallelize_cv
        self.clf = Pipeline([
            ('gp_weights', None),
            ('scale', None),
            ('norm', None),
            (self.name, clf),
        ])

        # try with and without scaling and normalization
        pg = {
            'gp_weights': [
                None,
                FunctionTransformer(
                    weight_multiplier,
                    kw_args={'fm': self, 'weight': 'score'}),
                FunctionTransformer(
                    weight_multiplier,
                    kw_args={'fm': self, 'weight': 'f_measure'}),
                FunctionTransformer(
                    weight_multiplier_gpprec,
                    kw_args={'fm': self}),
            ],
            'scale': [None, StandardScaler()],
            'norm': [None, Normalizer()],
        }
        if param_grid:
            # transform params to pipeline
            pg.update({
                '%s__%s' % (self.name, k): v
                for k, v in param_grid.items()
            })
        self.param_grid = ParameterGrid(pg)
        self.grid_search_top_results = None

        self.model = None  # None until trained, then best_estimator or clf
        self._fuse_auto_load = True
        self.loaded = False

    def save(self, filename=None, overwrite=False):
        if not self.loaded or overwrite:
            save_fusion_model(filename, self, overwrite)

    def load(self, filename=None):
        res = load_fusion_model(filename, self)
        if res:
            if self.gps is None:
                self.gps = res.gps
            if self.gps == res.gps:
                self.clf = res.clf
                self.model = res.model
                self.grid_search_top_results = res.grid_search_top_results
                self.loaded = True
                logger.info(
                    'loaded model %s with params: %s',
                    self.name, self.model.get_params())
            else:
                logger.warning(
                    'ignoring loading attempt for fusion model %s that seems '
                    'to have been trained on other gps',
                    self.name
                )
        return self.loaded

    def predict_scores(self, vecs, clf=None):
        if clf is None:
            clf = self.clf
        g_preds = clf.predict_proba(vecs)
        g_scores = np.sum(g_preds * clf.classes_, axis=1)
        return g_scores

    def scores(self, vecs, labels, groups, clf=None):
        """Calculates (test-set) avg precisions based on gtp groups."""
        if clf is None:
            clf = self.clf
        res = []
        gtp_ids = set(groups)
        for gtp_id in gtp_ids:
            g_mask = groups == gtp_id
            g_test_vecs = vecs[g_mask]
            g_test_labels = labels[g_mask]
            assert np.sum(g_test_labels) <= 1 or config.FUSION_PARAM_TUNING, \
                'expected only one true label per gtp (ground truth _pair_), ' \
                'but got %s for gtp_id %d' % (np.sum(g_test_labels), gtp_id)

            # get predictions for this group and sort labels by it
            g_scores = self.predict_scores(g_test_vecs, clf)
            ranked_labels = np.array([
                l for l, p in
                sorted(
                    zip(g_test_labels, g_scores),
                    key=itemgetter(1),  # intentionally ignore label
                    reverse=True,
                )
            ])
            relevant = np.sum(ranked_labels)
            assert relevant <= 1 or config.FUSION_PARAM_TUNING
            if relevant > 0:
                avg_prec = np.sum(
                    ranked_labels
                    / np.arange(1, len(ranked_labels) + 1)
                ) / relevant
            else:
                avg_prec = 0
            res.append(avg_prec)
        return res

    def preprocess_training_data(self, vecs, labels, groups):
        if config.FUSION_CMERGE_VECS or self.train_on_ratios:
            # Without merging, we typically have to deal with ~60K vecs.
            # With merging ~5K vecs.
            # This allows us to also train "slow" classifiers (such as KNN or
            # SVMs despite quadratic runtime).
            #
            # The following massively speeds up training of classifiers by
            # merging all vectors that occur multiple times (maybe even with
            # different ground truth labels). Multi occurrence actually happens
            # quite often when a single pattern is noisy and creates hundreds of
            # target candidates that aren't predicted by any other pattern. All
            # corresponding vectors will be the same (1 for this pattern, 0 for
            # all others). Even if the actual ground truth target is among these
            # candidates, the resulting vectors are mostly noise and convey very
            # little information.
            #
            # We treat the merged label as True only if the ratio of True labels
            # within all labels for the merged vector is above a certain
            # threshold. A threshold of 20 % for example means that the fusion
            # classifier might learn to falsely classify 4 wrong candidates as
            # true label. Considering the heavily imbalanced training vector
            # data (< 1/100 true positives) and the fact that we are actually
            # using class probabilities for ranking later on, it seems
            # reasonable that a small number of false positives is better than
            # losing this indicator altogether.
            # In accordance with this, experiments show that the effect is in
            # general quite positive on all classifiers except for qda.
            # Effects: In general positive on all classifiers but qda. Neural
            # net gains recall@2 but loses minor recall@10.
            vecs_, ratios = vecs_labels_to_unique_vecs_ratio(vecs, labels)
            labels = ratios >= config.FUSION_CMERGE_VECS_R

            # merging vectors is problematic wrt. gtps which we use as
            # "group" for GroupKFold. The following is a simplification to
            # still have the above benefits, by assigning the full vector
            # randomly to one of its gtps.
            vec_gtpidxs = defaultdict(list)
            for vec, gtpidx in zip(vecs, groups):
                vec_gtpidxs[tuple(vec)].append(gtpidx)
            r = random.Random(42)
            groups = np.array([
                r.choice(vec_gtpidxs[tuple(v)]) for v in vecs_])

            vecs = vecs_

            if self.train_on_ratios:
                # for example for regressors
                labels = ratios

        if config.FUSION_SAMPLES_PER_CLASS > 0:
            n = config.FUSION_SAMPLES_PER_CLASS
            idxs_true = np.where(labels > 0)[0]
            idxs_false = np.where(labels == 0)[0]
            r = random.Random(42)
            idxs_true = r.sample(idxs_true, min(len(idxs_true), n))
            idxs_false = r.sample(idxs_false, min(len(idxs_false), n))
            idxs = list(idxs_true) + list(idxs_false)
            r.shuffle(idxs)
            vecs, labels, groups = vecs[idxs], labels[idxs], groups[idxs]

        return vecs, labels, groups

    def train(self, gps, gtps, target_candidate_lists, training_tuple=None):
        self.gps = gps
        self.load()
        if self.model:
            logger.info(
                're-using trained fusion model %s from previous exec',
                self.name)
            return
        if not training_tuple:
            training_tuple = prep_training(gps, gtps, target_candidate_lists)
        vecs, labels, vtcs, vgtps, groups = training_tuple

        vecs, labels, groups = self.preprocess_training_data(
            vecs, labels, groups)

        # noinspection PyTypeChecker
        logger.info(
            '%s: training fusion model on %d samples, %d features: '
            '(pos: %d, neg: %d)',
            self.name, vecs.shape[0], vecs.shape[1],
            np.sum(labels > 0), np.sum(labels == 0)
        )
        score = 0
        if config.FUSION_PARAM_TUNING:
            # optimize MAP over splits on gtp groups
            param_candidates = list(self.param_grid)
            logger.info(
                '%s: performing grid search on %d parameter combinations',
                self.name, len(param_candidates)
            )

            score_func = partial(
                crossval_fm_scores,
                fm=self, vecs=vecs, labels=labels, groups=groups,
                n_params=len(param_candidates),
            )
            map_ = parallel_map if self.parallelize_cv else map
            scores = map_(score_func, enumerate(param_candidates, 1))

            top_score_candidates = sorted(
                zip(scores, param_candidates), key=itemgetter(0), reverse=True,
            )[:10]
            self.grid_search_top_results = top_score_candidates[:10]
            score, best_params = top_score_candidates[0]

            if score == -1:
                logger.warning(
                    '%s: grid search done, but each parameter combination '
                    'caused an exception. Ignoring this fusion model.',
                    self.name
                )
            else:
                logger.info(
                    '%s: grid search done, results for top 10 params:\n%s\n'
                    're-fitting on full train set with best params...',
                    self.name,
                    "\n".join([
                        '  %2d. mean score (MAP): %.3f (std: %.3f): %s' % (
                            i, mean, std, params)
                        for i, ((mean, std), params) in enumerate(
                            top_score_candidates[:10], 1)
                    ])
                )
                # refit classifier with best params
                self.clf.set_params(**best_params)
        if score > -1:
            # will be set to -1 if all params caused errors
            # noinspection PyBroadException
            try:
                self.clf.fit(vecs, labels)
                self.model = self.clf

                score = np.mean(self.scores(vecs, labels, groups))
                logger.info('%s: MAP on training set: %.3f', self.name, score)
                self.save()
            except Exception:
                logger.warning(
                    "Training %s caused an exception. "
                    "Ignoring this fusion model.",
                    exc_info=1
                )

    def fuse(self, gps, target_candidate_lists, targets_vecs=None):
        if not self.model and self._fuse_auto_load:
            self.load()
            if not self.model:
                logger.warning(
                    'fusion model for %s could not be loaded, did you run '
                    'train first?',
                    self.name
                )
        if not self.model:
            return []
        self._fuse_auto_load = False
        if targets_vecs:
            targets, vecs = targets_vecs
        else:
            targets, vecs = gp_tcs_to_vecs(gps, target_candidate_lists)
        if not targets:
            return []
        scores = self.predict_scores(vecs)
        return sorted(zip(targets, scores), key=itemgetter(1), reverse=True)


classifier_fm_slow = [
    # maybe improve performance by just changing n_neighbors after fitting?
    # training & prediction takes long on unfused vecs:
    FusionModel(
        "knn",
        KNeighborsClassifier(leaf_size=200),
        param_grid={
            'n_neighbors': [1, 2, 3, 4, 5, 8, 16, 32, 64, 128],
            'weights': ['uniform', 'distance']
        },
    ),
    # training takes long:
    FusionModel(
        "svm_linear",
        SVC(
            kernel='linear',
            probability=True,
            max_iter=config.FUSION_SVM_MAX_ITER,
        ),
        param_grid={
            'C': np.logspace(-5, 15, 11, base=2),
            'class_weight': ['balanced', None],
        },
    ),
    # training takes long:
    FusionModel(
        "svm_rbf",
        SVC(
            kernel='rbf',
            probability=True,
            max_iter=config.FUSION_SVM_MAX_ITER,
        ),
        param_grid={
            'C': np.logspace(-5, 15, 11, base=2),
            'gamma': np.logspace(-15, 3, 10, base=2),
            'class_weight': ['balanced', None]
        },
    ),
    # # training out of ram:
    # FusionModel(
    #     "gaussian_process",
    #     GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    # ),
]

classifier_fm_fast = [
    FusionModel(
        "decision_tree",
        DecisionTreeClassifier(),
        param_grid={
            'max_depth': [2, 5, 10, None],
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_features': [10, 'auto', 'log2', None],
            'class_weight': ['balanced', None],
        },
    ),
    FusionModel(
        "random_forest",
        RandomForestClassifier(),
        param_grid={
            'max_depth': [2, 5, 10, None],
            'n_estimators': [2, 5, 10, 15, 20, 25],
            'max_features': [10, 'auto', 'log2', None],
            'class_weight': ['balanced', None],
        },
    ),
    FusionModel(
        "gtb",
        GradientBoostingClassifier(),
    ),
    FusionModel(
        "adaboost",
        AdaBoostClassifier(),
    ),
    FusionModel(
        "neural_net",
        MLPClassifier(max_iter=1000),
        param_grid={
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
            'hidden_layer_sizes': [
                (10,), (15,), (20,), (25,), (30,), (50,), (75,), (100,),
                (10, 10), (15, 15), (20, 20), (25, 25), (50, 50), (100, 100),
                (100,), (50,), (10,), (10, 10), (100, 100),
            ],
        },
        # parallelize_cv=False,  # see env vars in run_create_bundle.sh
    ),
    FusionModel(
        "naive_bayes",
        GaussianNB(),
    ),
    FusionModel(
        "qda",
        QuadraticDiscriminantAnalysis(),
    ),
    FusionModel(
        'sgd',
        SGDClassifier(loss='log'),
        param_grid={
            'loss': ['log', 'modified_huber'],
            'class_weight': ['balanced', None]
        },
        # parallelize_cv=False,  # see env vars in run_create_bundle.sh
    ),
    FusionModel(
        'logistic_regression',
        LogisticRegression(),
    ),
]

classifier_fm = classifier_fm_slow + classifier_fm_fast


class FusionRegressionModel(FusionModel):
    train_on_ratios = True

    def predict_scores(self, vecs, clf=None):
        if clf is None:
            clf = self.clf
        g_scores = clf.predict(vecs)
        return g_scores


regression_fm_slow = [
    FusionRegressionModel(
        'kneighbors_r',
        KNeighborsRegressor(leaf_size=200),
        param_grid={
            'n_neighbors': [1, 2, 4, 8, 16, 32, 64, 128],
            'weights': ['uniform', 'distance']
        },
    ),
    FusionRegressionModel(
        'svr_linear',
        SVR(kernel='linear', max_iter=config.FUSION_SVM_MAX_ITER),
        param_grid={
            'C': np.logspace(-5, 15, 11, base=2),
        },
    ),
    FusionRegressionModel(
        'svr_rbf',
        SVR(kernel='linear', max_iter=config.FUSION_SVM_MAX_ITER),
        param_grid={
            'C': np.logspace(-5, 15, 11, base=2),
            'gamma': np.logspace(-15, 3, 10, base=2),
        },
    ),
]

regression_fm_fast = [
    FusionRegressionModel(
        "decision_tree_r",
        DecisionTreeRegressor(),
        param_grid={
            'max_depth': [2, 5, 10, None],
            'criterion': ['mse', 'mae'],
            'splitter': ['best', 'random'],
            'max_features': [10, 'auto', 'log2', None],
        },
    ),
    FusionRegressionModel(
        'ada_boost_r',
        AdaBoostRegressor(),
    ),
    FusionRegressionModel(
        'gradient_boosting_r',
        GradientBoostingRegressor(),
        param_grid={},
    ),
    FusionRegressionModel(
        'random_forest_r',
        RandomForestRegressor(),
        param_grid={
            'max_depth': [2, 5, 10, None],
            'n_estimators': [2, 5, 10, 15, 20, 25],
            'max_features': [10, 'auto', 'log2', None],
        },
    ),
    FusionRegressionModel(
        'kernel_ridge',
        KernelRidge(),
    ),
    FusionRegressionModel(
        'ard_r',
        ARDRegression(),
    ),
    FusionRegressionModel(
        'bayesian_ridge',
        BayesianRidge(),
    ),
    FusionRegressionModel(
        'elastic_net',
        ElasticNet(),
    ),
    FusionRegressionModel(
        'lars',
        Lars(),
    ),
    FusionRegressionModel(
        'lasso',
        Lasso(),
    ),
    FusionRegressionModel(
        'lasso_lars',
        LassoLars(),
    ),
    FusionRegressionModel(
        'linear_r',
        LinearRegression(),
    ),
    FusionRegressionModel(
        'ridge',
        Ridge(),
    ),
    FusionRegressionModel(
        'sgd_r',
        SGDRegressor(),
    ),
    FusionRegressionModel(
        'mlp_r',
        MLPRegressor(),
        param_grid={
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
            'hidden_layer_sizes': [
                (10,), (15,), (20,), (25,), (30,), (50,), (75,), (100,),
                (10, 10), (15, 15), (20, 20), (25, 25), (50, 50), (100, 100),
                (100,), (50,), (10,), (10, 10), (100, 100),
            ],
        },
    ),
]

regression_fm = regression_fm_slow + regression_fm_fast


class RankSVMFusionModel(FusionModel):
    name = 'RankSVMFusion'

    def predict_scores(self, vecs, clf=None):
        if clf is None:
            clf = self.clf
        return clf.decision_function(vecs)


ranksvm_fm = [
    RankSVMFusionModel(
        'rank_svm',
        RankSVM(),
        param_grid={
            'C': np.logspace(-5, 15, 11, base=2),
            'class_weight': ['balanced', None],
        },
    )
]


# TODO: more learning to rank methods?
# https://github.com/ogrisel/notebooks/blob/master/Learning%20to%20Rank.ipynb
