# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
from collections import OrderedDict
from operator import mul, itemgetter
import logging

import numpy as np
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

    def train(self, gps, gtps, target_candidate_lists, vecs_labels=None):
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


_basic_fusion_methods = [
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
basic_fusion_methods = OrderedDict([
    (_fm.name, _fm) for _fm in _basic_fusion_methods
])


def candidate_lists_to_gp_vectors(
        _, gtps, target_candidate_lists,
        print_vecs=False,
        warn_about_multiclass_vecs=False,
):
    assert len(gtps) == len(target_candidate_lists)
    logger.info('transforming all candidate lists to vectors')
    vecs = []
    labels = []
    for gtp, gp_tcs in zip(gtps, target_candidate_lists):
        source, target = gtp
        targets = set(tc for tcs in gp_tcs for tc in tcs)
        for t in targets:
            vec = tuple(t in tcs for tcs in gp_tcs)
            label = t == target
            vecs.append(vec)
            labels.append(label)

    if print_vecs:
        for v, l in zip(vecs, labels):
            print('%s: %s' % (l, [1 if x else 0 for x in v]))

    if warn_about_multiclass_vecs:
        # warn about vectors occurring several times with different labels
        c = Counter(vecs)
        logger.info('unique vectors: %d, (total: %d)', len(c), len(vecs))
        for v, occ in c.most_common():
            if occ == 1:
                break
            idxs = [i for i, vec in enumerate(vecs) if vec == v]
            cl = Counter([labels[i] for i in idxs])
            if len(cl) > 1:
                logger.warning(
                    'same vector shall be classified differently: %s\n%s',
                    cl, [1 if x else 0 for x in v]
                )

    return np.array(vecs), np.array(labels)


def gp_tcs_to_vecs(_, gp_tcs):
    targets = sorted(set(tc for tcs in gp_tcs for tc in tcs))
    vecs = []
    for t in targets:
        vec = tuple(t in tcs for tcs in gp_tcs)
        vecs.append(vec)
    return targets, np.array(vecs)


class FusionModel(object):
    def __init__(self, name, clf):
        self.name = name
        self.gps = None
        self.clf = clf
        self.model = None  # None until trained, then clf
        self._fuse_auto_load = True
        self.loaded = False

    def train(self, gps, gtps, target_candidate_lists, vecs_labels=None):
        self.gps = gps
        self.load()
        if self.model:
            logger.info(
                're-using trained fusion model %s from previous exec',
                self.name)
            return
        if vecs_labels:
            vecs, labels = vecs_labels
        else:
            vecs, labels = candidate_lists_to_gp_vectors(
                gps, gtps, target_candidate_lists)
        logger.info(
            'training fusion model %s on %d samples', self.name, len(vecs))
        self.clf.fit(vecs, labels)
        self.model = self.clf
        score = self.model.score(vecs, labels)
        logger.info('score on training set: %.3f', score)
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
                self.model = res.model
                self.clf = self.model
                self.loaded = True
            else:
                logger.warning(
                    'ignoring loading attempt for fusion model %s that seems '
                    'to have been trained on other gps',
                    self.name
                )
        return self.loaded

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
        # pred = self.model.predict(vecs)
        pred_class_probs = self.model.predict_proba(vecs)
        # our classes are boolean, [0,1], get probs for [1] by * and sum
        probs = np.sum(pred_class_probs * self.model.classes_, axis=1)
        return sorted(zip(targets, probs), key=itemgetter(1, 0), reverse=True)


_fm = FusionModel

_classifier_fusion_methods = [
    # training & prediction takes long:
    FusionModel(
        "knn3",
        KNeighborsClassifier(3)),
    # training & prediction takes long:
    FusionModel(
        "knn5",
        KNeighborsClassifier(5)),
    FusionModel(
        "svm_linear",
        SVC(kernel="linear", C=0.1, probability=True, class_weight='balanced')),
    # training takes long:
    FusionModel(
        "svm_rbf",
        SVC(gamma=2, C=1, probability=True, class_weight='balanced')),
    # # training out of ram:
    # FusionModel(
    #     "gaussian_process",
    #     GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)),
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
classifier_fusion_methods = OrderedDict([
    (_fm.name, _fm) for _fm in _classifier_fusion_methods
])

# class RankSVMFusion(FusionModel):
#     name = 'rank_svm'
#
# _fusion_methods['rank_svm'] = RankSVMFusion()

all_fusion_methods = OrderedDict(
    basic_fusion_methods.items()
    + classifier_fusion_methods.items()
)


def get_fusion_methods_from_str(fms_arg=None):
    if not fms_arg:
        return all_fusion_methods.values()

    fmsl = [s.strip() for s in fms_arg.split(',')]
    # replace with fusion methods, also expanding 'basic' and 'classifiers'
    fml = []
    for s in fmsl:
        if s == 'basic':
            fml.extend(basic_fusion_methods.values())
        elif s == 'classifiers':
            fml.extend(classifier_fusion_methods.values())
        else:
            try:
                fml.append(all_fusion_methods[s])
            except KeyError:
                logger.error(
                    'unknown fusion method: %s\navailable: %s',
                    s, ['basic', 'classifiers'] + all_fusion_methods.keys()
                )
                raise

    return fml


def train_fusion_models(gps, gtps, target_candidate_lists, fusion_methods=None):
    vecs_labels = candidate_lists_to_gp_vectors(
        gps, gtps, target_candidate_lists)
    for fm in get_fusion_methods_from_str(fusion_methods):
        fm.train(gps, gtps, target_candidate_lists, vecs_labels)


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
