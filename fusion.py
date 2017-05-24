# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
from collections import OrderedDict
from operator import mul, itemgetter
import logging

from sklearn import svm
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


from serialization import load_fusion_model
from serialization import save_fusion_model

logger = logging.getLogger(__name__)


class Fusion(object):
    name = 'base_fusion'

    def train(self, gps, gtps, target_candidate_lists):
        pass

    def save(self, filename=None):
        pass

    def load(self, filename=None):
        pass

    def fuse(self, gps, target_candidate_lists):
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


basic_fusion_methods = [
    BasicWeightedFusion(
        'target_occs',
        'a simple occurrence count of the target over all gps.'),
    BasicWeightedFusion(
        'scores',
        'sum of all gp scores for each returned target.',
        getter_gp=_score_getter),
    BasicWeightedFusion(
        'f_measures',
        'sum of all gp f_measures for each returned target.',
        getter_gp=_f_measure_getter),
    BasicWeightedFusion(
        'gp_precisions',
        'sum of all gp precisions for each returned target.',
        getter_gp=_gp_precisions_getter),
    BasicWeightedFusion(
        'precisions',
        'sum of the actual precisions per gp in this prediction.',
        getter_tcs=_precisions_getter),
    BasicWeightedFusion(
        'scores_precisions',
        'same as above but scaled with precision',
        getter_gp=_score_getter, getter_tcs=_precisions_getter),
    BasicWeightedFusion(
        'f_measures_precisions',
        'same as above but scaled with precision',
        getter_gp=_f_measure_getter, getter_tcs=_precisions_getter),
    BasicWeightedFusion(
        'gp_precisions_precisions',
        'same as above but scaled with precision',
        getter_gp=_gp_precisions_getter, getter_tcs=_precisions_getter),
]


def candidate_lists_to_gp_vectors(gps, gtps, target_candidate_lists):
    assert len(gtps) == len(target_candidate_lists)
    X = []
    y = []
    for gtp, gp_tcs in zip(gtps, target_candidate_lists):
        source, target = gtp
        targets = set(tc for tcs in gp_tcs for tc in tcs)
        for t in targets:
            vec = [t in tcs for tcs in gp_tcs]
            label = t == target
            X.append(vec)
            y.append(label)
    return X, y


def gp_tcs_to_vecs(gps, gp_tcs):
    targets = sorted(set(tc for tcs in gp_tcs for tc in tcs))
    X = []
    for t in targets:
        vec = [t in tcs for tcs in gp_tcs]
        X.append(vec)
    return targets, X


class FusionModel(object):
    def __init__(self, name, clf):
        self.name = name
        self.clf = clf
        self.model = None  # None until trained, then clf
        self._fuse_auto_load = True

    def train(self, gps, gtps, target_candidate_lists):
        logger.info('training fusion model %s', self.name)
        X, y = candidate_lists_to_gp_vectors(gps, gtps, target_candidate_lists)
        self.clf.fit(X, y)
        self.model = self.clf
        score = self.model.score(X, y)
        logger.info('score on training set: %.3f', score)

    def save(self, filename=None):
        logger.info('saving fusion model %s', self.name)
        save_fusion_model(filename, self.name, self.model)

    def load(self, filename=None):
        logger.info('loading fusion model %s', self.name)
        self.model = load_fusion_model(filename, self.name)

    def fuse(self, gps, target_candidate_lists):
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
        targets, X = gp_tcs_to_vecs(gps, target_candidate_lists)
        pred = self.model.predict(X)
        return sorted(zip(targets, pred), key=itemgetter(1, 0), reverse=True)


_fm = FusionModel

classifiers = [
    FusionModel(
        "knn3",
        KNeighborsClassifier(3)),
    FusionModel(
        "knn5",
        KNeighborsClassifier(5)),
    FusionModel(
        "svm_linear",
        SVC(kernel="linear", C=0.025)),
    FusionModel(
        "svm_rbf",
        SVC(gamma=2, C=1)),
    FusionModel(
        "gaussian_process",
        GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)),
    FusionModel(
        "decision_tree",
        DecisionTreeClassifier(max_depth=5)),
    FusionModel(
        "random_forest",
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)),
    FusionModel(
        "neural_net",
        MLPClassifier(alpha=1)),
    FusionModel(
        "adaboost",
        AdaBoostClassifier()),
    FusionModel(
        "naive_bayes",
        GaussianNB()),
    FusionModel(
        "qda",
        QuadraticDiscriminantAnalysis()),
]

# class RankSVMFusion(FusionModel):
#     name = 'rank_svm'
#
# _fusion_methods['rank_svm'] = RankSVMFusion()



_fusion_methods = OrderedDict([
    (_fm.name, _fm) for _fm in basic_fusion_methods + classifiers
])


def train_fusion_models(gps, gtps, target_candidate_lists, fusion_methods=None):
    if fusion_methods is None:
        fusion_methods = _fusion_methods.values()
    else:
        assert fusion_methods and isinstance(fusion_methods[0], basestring)
        fusion_methods = [_fusion_methods[fmn] for fmn in fusion_methods]
    for fm in fusion_methods:
        fm.train(gps, gtps, target_candidate_lists)
        fm.save()


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

    res = OrderedDict()
    for fm in fusion_methods:
        try:
            res[fm.name] = fm.fuse(gps, target_candidate_lists)
        except NotImplementedError:
            logger.warning(
                'seems %s is not implemented yet, but called', fm.name
            )
            res[fm.name] = []
    return res
