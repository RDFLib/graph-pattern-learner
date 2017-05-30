# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import random
from collections import Counter
from collections import OrderedDict
from collections import defaultdict
from operator import mul, itemgetter
from functools import partial

import numpy as np
from scoop.futures import map as parallel_map
from sklearn import clone
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier

import config
from serialization import load_fusion_model
from serialization import save_fusion_model

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
    vtc = []
    vgtp = []
    vgtp_idxs = []
    labels = []
    for (gtp_idx, gtp), gp_tcs in zip(enumerate(gtps), target_candidate_lists):
        source, target = gtp
        targets = set(tc for tcs in gp_tcs for tc in tcs)
        for t in targets:
            vec = tuple(t in tcs for tcs in gp_tcs)
            label = t == target
            vecs.append(vec)
            vtc.append(t)
            vgtp.append(gtp)
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
    return a(vecs, dtype='f8'), a(labels), vtc, vgtp, a(vgtp_idxs)


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


def avg_precs(clf, vecs, labels, groups):
    """Calculates (test-set) avg precisions based on gtp groups.

    Called during training.
    """
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
        g_preds = clf.predict_proba(g_test_vecs)
        g_scores = np.sum(g_preds * clf.classes_, axis=1)
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


def score_map_clf_cv(enum_params, name, clf, vecs, labels, groups):
    """Calculates the mean avg precision score for a given classifier via CV.

    Used at training time to find the MAP for the given classifier and params
    via cross validation.
    """
    params_number, params = enum_params
    logger.debug('training classifier %s with params: %s', name, params)
    clf = clone(clf)
    clf.set_params(**params)
    n_splits = config.FUSION_PARAM_TUNING_CV_KFOLD
    splits = GroupKFold(n_splits).split(vecs, labels, groups=groups)
    all_avg_precs = []
    for split, (train_idxs, test_idxs) in enumerate(splits, 1):
        logger.debug('%s CV split %d/%d fitting...', name, split, n_splits)
        clf.fit(vecs[train_idxs], labels[train_idxs])
        logger.debug('%s CV split %d/%d fitted.', name, split, n_splits)

        # calculate test-set avg precisions based on gtp groups
        test_vecs = vecs[test_idxs]
        test_labels = labels[test_idxs]
        test_groups = groups[test_idxs]
        all_avg_precs.extend(
            avg_precs(clf, test_vecs, test_labels, test_groups))

    if params_number % 10 == 0:
        logger.info('completed crossval of param %d', params_number)
    mean_avg_prec = np.mean(all_avg_precs)
    std_avg_prec = np.std(all_avg_precs)
    return mean_avg_prec, std_avg_prec


class FusionModel(Fusion):
    def __init__(self, name, clf, param_grid=None):
        self.name = name
        self.gps = None
        self.clf = Pipeline([
            # TODO: maybe use FunctionTransformer to get gp dims?
            # ('gp_weights', FunctionTransformer()),
            ('scale', StandardScaler()),
            ('norm', Normalizer()),
            (self.name, clf),
        ])

        # try with and without scaling and normalization
        pg = {
            'scale': [None, StandardScaler()],
            'norm': [Normalizer(), None],
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

        if config.FUSION_CMERGE_VECS:
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

            if config.FUSION_PARAM_TUNING:
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

        logger.info(
            'training fusion model %s on %d samples, %d features: '
            '(pos: %d, neg: %d)',
            self.name, vecs.shape[0], vecs.shape[1],
            np.sum(labels), np.sum(labels < 1)
        )
        if config.FUSION_PARAM_TUNING:
            # optimize MAP over splits on gtp groups
            param_candidates = list(self.param_grid)
            logger.info(
                'performing grid search on %d parameter combinations',
                len(param_candidates)
            )
            score_func = partial(
                score_map_clf_cv,
                clf=self.clf, name=self.name,
                vecs=vecs, labels=labels, groups=groups,
            )
            scores = parallel_map(score_func, enumerate(param_candidates, 1))
            top_score_candidates = sorted(
                zip(scores, param_candidates), key=itemgetter(0), reverse=True,
            )[:10]
            _, best_params = top_score_candidates[0]
            logger.info(
                'grid search results for %s:\nTop 10 params:\n%s',
                self.name,
                "\n".join([
                    '%2d. mean score (MAP): %.3f (std: %.3f): %s' % (
                        i, mean, std, params)
                    for i, ((mean, std), params) in enumerate(
                        top_score_candidates[:10], 1)
                ])
            )
            self.grid_search_top_results = top_score_candidates[:10]
            # refit classifier with best params
            self.clf.set_params(**best_params)
            self.clf.fit(vecs, labels)
            self.model = self.clf
        else:
            self.clf.fit(vecs, labels)
            self.model = self.clf
        score = np.mean(avg_precs(self.clf, vecs, labels, groups))
        logger.info('MAP on training set: %.3f', score)
        self.save()

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

    def fuse(self, gps, target_candidate_lists, targets_vecs=None, clf=None):
        if not clf:
            clf = self.model
        if not clf and self._fuse_auto_load:
            self.load()
            if not self.model:
                logger.warning(
                    'fusion model for %s could not be loaded, did you run '
                    'train first?',
                    self.name
                )
        if not clf:
            return []
        self._fuse_auto_load = False
        if targets_vecs:
            targets, vecs = targets_vecs
        else:
            targets, vecs = gp_tcs_to_vecs(gps, target_candidate_lists)
        # pred = self.model.predict(vecs)
        pred_class_probs = clf.predict_proba(vecs)
        # our classes are boolean, [0,1], get probs for [1] by * and sum
        probs = np.sum(pred_class_probs * clf.classes_, axis=1)
        return sorted(zip(targets, probs), key=itemgetter(1), reverse=True)


classifier_fm_slow = [
    # maybe improve performance by just changing n_neighbors after fitting?
    # training & prediction takes long on unfused vecs:
    FusionModel(
        'knn3',
        KNeighborsClassifier(3, leaf_size=500, algorithm='ball_tree'),
        param_grid={'weights': ['uniform', 'distance']},
    ),
    # training & prediction takes long on unfused vecs:
    FusionModel(
        "knn5",
        KNeighborsClassifier(5, leaf_size=500, algorithm='ball_tree'),
        param_grid={'weights': ['uniform', 'distance']},
    ),
    # training takes long:
    FusionModel(
        "svm_linear",
        SVC(kernel='linear', probability=True),
        param_grid={
            'C': np.logspace(-5, 15, 11, base=2),
            'class_weight': ['balanced', None],
        },
    ),
    # training takes long:
    FusionModel(
        "svm_rbf",
        SVC(kernel='rbf', probability=True),
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
            # 'n_estimators': [2, 5, 10, 15, 20, 25],
            # 'max_features': [10, 'auto', 'log2', None],
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
        MLPClassifier(max_iter=300),
        param_grid={
            'alpha': [1, 0.01, 0.0001],
            # 'alpha': [1, 0.1, 0.01, 0.001, 0.0001],
            'hidden_layer_sizes': [
                # (10,), (15,), (20,), (25,), (30,), (50,), (75,), (100,),
                # (10, 10), (15, 15), (20, 20), (25, 25), (50, 50), (100, 100),
                (10,), (50,), (100,), (10, 10), (100, 100),
            ],
        },
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
        SGDClassifier(),
        param_grid={
            'loss': ['log', 'modified_huber'],
            'class_weight': ['balanced', None]
        },
    ),
    FusionModel(
        'bgmm',
        BayesianGaussianMixture(),
        param_grid={
            'n_components': range(1, 11)
        },
    ),
    FusionModel(
        'gmm',
        GaussianMixture(),
        param_grid={
            'n_components': range(1, 11)
        },
    ),
]

classifier_fm = classifier_fm_slow + classifier_fm_fast


# class FusionRegressionModel(FusionModel):
#     pass
#
#
# regression_fm = [
#     FusionRegressionModel(
#         'bgmm',
#         BayesianGaussianMixture(),
#         param_grid={
#             'n_components': range(1, 11)
#         },
#     ),
#     FusionRegressionModel(
#         'gmm',
#         GaussianMixture(),
#         param_grid={
#             'n_components': range(1, 11)
#         },
#     ),
#     # KernelDensity...
# ]


# class RankSVMFusion(FusionModel):
#     name = 'rank_svm'
#
#  _fusion_methods['rank_svm'] = RankSVMFusion()



# noinspection PyTypeChecker
all_fusion_methods = OrderedDict(
    (_fm.name, _fm) for _fm in
    basic_fm + classifier_fm
)


def get_fusion_methods_from_str(fms_arg=None):
    if not fms_arg:
        return all_fusion_methods.values()

    fmsl = [s.strip() for s in fms_arg.split(',')]
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
