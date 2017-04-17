# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
from collections import OrderedDict
from collections import defaultdict
from itertools import cycle
import logging
import math
import pprint

from matplotlib import pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from gtp_scores import GTPScores

logger = logging.getLogger(__name__)


class Cluster(object):
    def __init__(self, name, gps, samples, max_k):
        """Represents a cluster result on gps and samples.

        :param name: to identify the clustering variant
        :param gps: graph patterns
        :param samples: raw (unscaled!) gtps precision vectors corresp. to gps
        :param max_k: max_k used to create clustering and default for
            get_labels. Important especially for non hierarchical clustering
            methods which need to know it for the actual clustering and not only
            when retrieving the labels.
        """
        assert len(gps) == len(samples)
        self.name = name
        self.gps = gps
        self.samples = samples
        self.max_k = max_k

    def get_labels(self, max_k=None):
        raise NotImplementedError(
            'Cluster.get_labels() needs to be overridden in subclasses.'
        )

    def get_gp_cluster_labels(self, max_k=None):
        labels = self.get_labels(max_k)
        gp_cluster_labels = [
            (gp, labels[i]) for i, gp in enumerate(self.gps)
        ]
        return gp_cluster_labels

    def get_representative_gps_labels(self, max_k=None):
        if not max_k:
            max_k = self.max_k
        return pick_top_gps_per_clusters(
            self.get_gp_cluster_labels(max_k), max_per_cluster=1, n=max_k
        )


class HierarchicalCluster(Cluster):
    def __init__(self, name, gps, samples, max_k,
                 cluster_hierarchy, cophenet_coeff):
        super(HierarchicalCluster, self).__init__(name, gps, samples, max_k)
        self.cluster_hierarchy = cluster_hierarchy
        self.cophenet_coeff = cophenet_coeff

    def get_labels(self, max_k=None):
        if not max_k:
            max_k = self.max_k
        assert max_k > 0
        return fcluster(self.cluster_hierarchy, max_k, criterion='maxclust')


def gp_precs_matrix(gps, gtps, plot_sample_len_hist=False):
    samples = [
        gp.get_gtps_precision_vector(gtps) for gp in gps
    ]
    prec_vec_count = Counter(samples)
    logger.info('distinct precision vectors: %d' % len(prec_vec_count))
    samples = np.array(samples)

    if plot_sample_len_hist:
        logger.info(samples.sum(axis=1))
        # noinspection PyTypeChecker
        plt.hist(np.linalg.norm(samples, ord=0, axis=1))  # != 0
        # noinspection PyTypeChecker
        plt.hist(np.linalg.norm(samples, ord=1, axis=1))  # sum
        plt.show()

    return samples


def make_cluster_variants_for_gps_by_precision(
        gps,
        gtps,
        max_k,
        variants=None,
        plot_dendrograms=False
):
    samples = gp_precs_matrix(gps, gtps)
    variants = _make_cluster_variants(gps, samples, max_k, variants)

    if plot_dendrograms:
        plot_cluster_variants_dendrograms(variants)

    return variants


def _make_cluster_variants(gps, samples, max_k, variants=None):
    res = {}
    # TODO: performance: don't recompute hierarchical clusterings for diff max_k

    # TODO: include sklearn hierarchical clustering
    # model = AgglomerativeClustering()
    # model.fit(samples)
    # print('clustering results:')
    # print('labels:')
    # print(model.labels_)
    # print('n_leaves:')
    # print(model.n_leaves_)
    # print('n_components:')
    # print(model.n_components_)
    # print('children:')
    # print(model.children_)

    # TODO: include non hierarchical variants
    # a difference of 0.2 in cosine similarity is allowed to merge clusters
    # model = AffinityPropagation()
    # model.fit(samples)
    # labels = model.labels_
    # core_samples_mask = np.zeros_like(labels, dtype=bool)
    # core_samples_mask[model.core_sample_indices_] = True

    metrics = ['euclidean', 'cityblock', 'cosine']
    methods = [
        'single', 'complete', 'weighted', 'average',
        'centroid', 'median', 'ward',
    ]

    for scale in ['', 'scaled_']:
        ssamples = samples
        if scale:
            ssamples = StandardScaler().fit_transform(samples)

        for metric in metrics:
            cdist = pdist(ssamples, metric)
            if metric == 'cosine':
                # see https://github.com/scipy/scipy/issues/5208
                np.clip(cdist, 0, 1, out=cdist)

            for method in methods:
                name = '%s%s_%s' % (scale, metric, method)
                logger.debug('computing clustering %s', name)
                try:
                    if variants and name not in variants:
                        # could skip earlier but would make code more complex
                        continue
                    if method in ['ward', 'centroid', 'median']:
                        # method needs raw feature vectors in euclidean space
                        if metric == 'euclidean':
                            cluster_hierarchy = linkage(ssamples, method=method)
                        else:
                            continue
                    elif method not in [
                            'single', 'complete', 'weighted', 'average']:
                        # method needs raw inputs, recompute:
                        if metric == 'cosine':
                            # see: https://github.com/scipy/scipy/issues/5208
                            continue
                        cluster_hierarchy = linkage(
                            ssamples, method=method, metric=metric)

                    else:
                        cluster_hierarchy = linkage(cdist, method=method)

                    c, coph_dists = cophenet(cluster_hierarchy, cdist)

                    res[name] = HierarchicalCluster(
                        name, gps, samples, max_k, cluster_hierarchy, c)
                    logger.info('clustering %s computed with c: %0.3f', name, c)
                except ValueError:
                    logger.warning(
                        'The following exception occurred during clustering '
                        'with variant %s:\nException:',
                        name,
                        exc_info=1,  # appends exception to message
                    )
    logger.info('computed %d clustering variants', len(res))
    return res


def print_cluster_stats(cluster, samples, max_k=None):
    assert isinstance(cluster, Cluster)
    labels = cluster.get_labels(max_k)
    label_counts = Counter(labels)

    print('Estimated number of clusters: %d' % len(label_counts))
    print('Cluster distributions (idx, count):')
    pprint.pprint(label_counts.most_common())
    print("Silhouette Coefficient: %0.3f" % silhouette_score(samples, labels))
    for i, l in enumerate(labels, 1):
        print('GraphPattern %d: %s' % (i, l))
    # for l in sorted(set(labels)):
    #     print('label %s' % l)
    #     for i, il in enumerate(labels):
    #         if il == l:
    #             print_graph_pattern(result_patterns[i][0], 0)
    #     print('\n\n\n-------------------\n\n\n')


def plot_cluster_variants_dendrograms(variants):
    # dendrogram doesn't make sense for non hierarchical clustering
    variants = {
        k: v for k, v in variants.items() if isinstance(v, HierarchicalCluster)
    }
    # get rows and cols in ~4:3 ratio
    n = len(variants)
    assert n > 0
    cols = (4 * n / 3)**.5
    rows = int(n // cols)
    cols = int(math.ceil(n / rows))
    fig, axes = plt.subplots(rows, cols)
    for i, (v_name, variant) in enumerate(sorted(variants.items()), 1):
        plt.subplot(rows, cols, i)
        plt.title(
            '%s, c: %.3f' % (v_name, variant.cophenet_coeff)
        )
        fancy_dendrogram(
            variant.cluster_hierarchy,
            truncate_mode='lastp',
            p=20,
            leaf_rotation=45.,
            leaf_font_size=8.,
            show_contracted=True,
            annotate_above=100,
        )

    for ax in axes.flatten()[n:]:
        fig.delaxes(ax)
    # plt.tight_layout()
    plt.show()


def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(
                ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


def pick_top_gps_per_clusters(
        gp_cluster_labels, max_per_cluster=None, n=None):
    """Picks from ordered patterns but round robins through clusters.

    Useful for example to pick from patterns ordered by desc fitness but rotate
    through clusters before picking very similar patterns.

    :param gp_cluster_labels: List of (gp, cluster_label). cluster_labels
        should be ints >= 0. Negative cluster_labels are interpreted as a rest
        class cluster as returned by DBSCAN.
    :param max_per_cluster: (optional) can be given to restrict returned gps per
        cluster. Again negative cluster_labels aren't affected by this.
    :param n: (optional) max desired length.
    :return: List of (gp, cluster_label) pairs.
    """
    assert max_per_cluster is None or max_per_cluster > 0
    res = []
    round_ = 1
    clusters_seen_in_round = set()
    backlog = list(gp_cluster_labels)
    if n is None:
        n = len(backlog)
    backlog.append((None, None))  # stop element
    while len(res) < n and backlog:
        gp, label = item = backlog[0]
        backlog = backlog[1:]
        if item == (None, None):
            round_ += 1
            clusters_seen_in_round = set()
            if backlog:
                # still other items (non stop element) in backlog
                backlog.append(item)
            else:
                break
        elif label in clusters_seen_in_round:
            # shift to next round
            if max_per_cluster is None or max_per_cluster > round_:
                backlog.append(item)
        else:
            # found the next item
            res.append(item)
            if label >= 0:
                clusters_seen_in_round.add(label)
    return res


def expected_precision_loss_by_query_reduction(
        gps, gtps, max_ks, gtp_scores,
        variants=None,
        plot_precision_losses_over_k=False):
    """Calculates the expected precision loss for k requests on training set.

    As the amount of found graph patterns `gps` to cover a given ground truth
    pair list `gtps` can be quite long and a lot of those patterns can be very
    similar to each other wrt. information gain, it might be desirable to make
    less than `len(gps)` requests when predicting. This method evaluates the
    lost overall score on our training data set `gtps` when reducing the amount
    of queries down to `max_k`. We try to be smart about the reduction by
    clustering the given `gps` by similarity and picking the best representative
    pattern in each cluster in order to retain as much of the overall
    information gains of the different queries (maximising variability).

    We already use a multitude of clustering methods as they behave quite
    different, but the list obviously can be expanded (also to non clustering
    algorithms). Also picking the "fittest" pattern as cluster representative
    might not be the best solution in all cases.

    We're dealing with an optimization variant of the NP-complete set cover
    problem here.

    :param gps: graph pattern list as returned by `find_graph_pattern_coverage`,
        just without the run number
    :param gtps: list of ground truth pairs used for training
    :param max_ks: list of possible `max_k` values for which we run the query
        reduction variants
    :param gtp_scores: as returned by `find_graph_pattern_coverage` to calculate
        the maximally possible score and the losses
    :param variants: list of reduction variants or `None` (default) for all
    :param plot_precision_losses_over_k: if set, will show a plot of decreasing
        precision loss over increasing `max_k`
    :return: `variant_max_k_prec_loss_reps`: nested dict like, indexed with the
        variant name and `max_k` (OrderedDict). Something like this:
        variant_max_k_prec_loss_reps[variant_name][k] = (prec_loss, reps)
    """
    max_score = gtp_scores.score
    variant_max_k_prec_loss_reps = defaultdict(OrderedDict)
    for max_k_idx, max_k in enumerate(max_ks):
        # TODO: try other (non clustering) algorithms to improve set coverage
        # TODO: picking fittest representative might be non-optimal solution
        cluster_variants = make_cluster_variants_for_gps_by_precision(
            gps, gtps, max_k, variants=variants)
        for cv_name, cv in sorted(cluster_variants.items()):
            gp_rep_labels = cv.get_representative_gps_labels()
            gp_reps = [gp for gp, _ in gp_rep_labels]

            cluster_gtp_scores = GTPScores(gtps)
            all_gtp_precisions = defaultdict(list)
            for gp in gp_reps:
                for gtp, gtp_precision in gp.gtp_precisions.items():
                    all_gtp_precisions[gtp] += [gtp_precision]
            for gtp, precs in all_gtp_precisions.items():
                cluster_gtp_scores.gtp_max_precisions[gtp] = max(precs)

            lost_precision = gtp_scores - cluster_gtp_scores
            logger.debug(
                'Precision loss for max %d requests in clustering %s: %0.3f',
                max_k, cv_name, lost_precision
            )
            variant_max_k_prec_loss_reps[cv_name][max_k] = \
                (lost_precision / max_score, gp_reps)

    log_msg = ['Best clusterings (least precision loss) per k requests:']
    for k in max_ks:
        log_msg += ['k = %d:' % k]
        tops = sorted([
            (ploss_reps[k][0], cn)
            for cn, ploss_reps in variant_max_k_prec_loss_reps.items()
        ])[:5]
        log_msg += ['  %0.3f %s' % (pl, cn) for pl, cn in tops]
    logger.info('\n'.join(log_msg))

    log_msg = ['Precision losses by clustering:']
    line_style_cycler = cycle(["-", "--", "-.", ":"])
    for cv_name, k_ploss_reps in sorted(variant_max_k_prec_loss_reps.items()):
        log_msg += [cv_name]
        for k, (prec_loss, _) in k_ploss_reps.items():
            log_msg += ['  max_k: %2d, loss: %0.3f' % (k, prec_loss)]
        if plot_precision_losses_over_k:
            plt.plot(
                [0] + max_ks,
                [1.] + [ploss for ploss, _ in k_ploss_reps.values()],
                next(line_style_cycler),
                label=cv_name
            )
    logger.info('\n'.join(log_msg))
    if plot_precision_losses_over_k:
        plt.xlabel('requests')
        plt.ylabel('precision loss ratio')
        plt.legend()
        plt.show()

    return variant_max_k_prec_loss_reps


def select_best_variant(variant_max_k_prec_loss_reps, log_top_k=1):
    top_vars = sorted([
        (prec_loss, k, var_name, reps)
        for var_name, k_ploss_reps in variant_max_k_prec_loss_reps.items()
        for k, (prec_loss, reps) in k_ploss_reps.items()
    ])[:log_top_k]
    prec_loss, k, vn, reps = top_vars[0]
    logger.info(
        'selected query reduction variant:\n'
        '%s' +
        ('\nbetter than these follow-ups:\n%s' if log_top_k > 1 else '%s'),
        ' -> precision loss: %0.3f with %d queries: %s' % (prec_loss, k, vn),
        '\n'.join([
            '    precision loss: %0.3f with %d queries: %s' % (_pl, _k, _vn)
            for _pl, _k, _vn, _ in top_vars[1:]
        ])
    )
    return prec_loss, k, vn, reps
