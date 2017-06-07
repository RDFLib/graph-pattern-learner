#!/usr/bin/env python2.7
# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import gzip
from itertools import chain
import json
import logging
from os import path
import random

import nose
from rdflib import URIRef
from rdflib.util import from_n3
import six

import config

WP_LINKER_RES_FILENAME = 'wp_linker_results.json.gz'

logger = logging.getLogger(__name__)


_wp_linker_results = None


def load_linker_results(fn=None):
    global _wp_linker_results
    if fn is None:
        fn = path.join(
            path.dirname(__file__), config.DATADIR, WP_LINKER_RES_FILENAME)
    if not _wp_linker_results:
        with gzip.open(fn) as f:
            _wp_linker_results = json.load(f)
    return _wp_linker_results


def get_verified_mappings():
    """Only returns results which are out of question verified (3 positive)."""
    wp_linker_results = load_linker_results()
    res = {}
    for hash_, mapping in six.iteritems(wp_linker_results['ratings']):
        ratings = mapping['ratings']
        if (
            'False' not in ratings and
            'Skip' not in ratings and
            ratings.get('True', 0) > 2
        ):
            res[hash_] = mapping
    return res


def split_mapping_training_test_set(mappings, split=0.1, seed=42):
    return map(dict, split_training_test_set(mappings.items(), split, seed))


def wiki_to_dbpedia_link(wikilink):
    return wikilink.replace(
        'http://en.wikipedia.org/wiki/', 'http://dbpedia.org/resource/', 1)


def get_dbpedia_links_from_mappings(mappings):
    links = set()
    for _, mapping in six.iteritems(mappings):
        for link_kind in ['stimulus_link', 'response_link']:
            link = mapping[link_kind]
            links.add(wiki_to_dbpedia_link(link))
    return sorted(links)


def get_dbpedia_pairs_from_mappings(mappings):
    pairs = set()
    for _, mapping in six.iteritems(mappings):
        stimulus_link = wiki_to_dbpedia_link(mapping['stimulus_link'])
        response_link = wiki_to_dbpedia_link(mapping['response_link'])
        pairs.add((stimulus_link, response_link))
    return sorted(pairs)


# noinspection PyPep8Naming
def URIRefify(links):
    return tuple([URIRef(l) for l in links])


def get_semantic_associations(fn=None, limit=None):
    if not fn:
        verified_mappings = get_verified_mappings()
        semantic_associations = get_dbpedia_pairs_from_mappings(
            verified_mappings)
        semantic_associations = [URIRefify(p) for p in semantic_associations]
    else:
        semantic_associations = []
        with gzip.open(fn) if fn.endswith('.gz') else open(fn) as f:
            # expects a file with one space separated pair of n3 encoded IRIs
            # per line
            r = csv.DictReader(
                f,
                delimiter=b' ',
                doublequote=False,
                escapechar=None,
                quoting=csv.QUOTE_NONE,
            )
            assert r.fieldnames == ['source', 'target']
            for i, row in enumerate(r):
                if limit and i >= limit:
                    break
                source = from_n3(row['source'].decode('UTF-8'))
                target = from_n3(row['target'].decode('UTF-8'))
                semantic_associations.append((source, target))
    return semantic_associations


def write_semantic_associations(associations, fn=None):
    if fn is None:
        fn = config.GT_ASSOCIATIONS_FILENAME
    with open(fn, 'w') as f:
        # writes a file with one space separated pair of n3 encoded IRIs
        # per line
        w = csv.DictWriter(
            f,
            fieldnames=('source', 'target'),
            delimiter=b' ',
            doublequote=False,
            escapechar=None,
            quoting=csv.QUOTE_NONE,
        )
        w.writeheader()
        for source, target in associations:
            w.writerow({
                'source': source.n3().encode('UTF-8'),
                'target': target.n3().encode('UTF-8'),
            })


def filter_node_pairs_split(train, test, variant):
    assert variant in config.SPLITTING_VARIANTS
    if variant == 'target_node_disjoint':
        train_target_nodes = {t for s, t in train}
        tmp = [(s, t) for s, t in test if t not in train_target_nodes]
        logger.info(
            'removed %d/%d pairs from test set because of overlapping target '
            'nodes with training set',
            len(test) - len(tmp), len(test)
        )
        test = tmp
    elif variant == 'node_disjoint':
        train_nodes = {n for np in train for n in np}
        tmp = [
            (s, t) for s, t in test
            if s not in train_nodes and t not in train_nodes
        ]
        logger.info(
            'removed %d/%d pairs from test set because of overlapping '
            'nodes with training set',
            len(test) - len(tmp), len(test)
        )
        test = tmp
    return train, test


@nose.tools.nottest
def split_training_test_set(associations, split=0.1, seed=42, variant='random'):
    return next(
        k_fold_cross_validation(associations, int(1 / split), seed, variant)
    )


def k_fold_cross_validation(associations, k, seed=42, variant='random'):
    """Generates k folds of train and validation sets out of associations.

    >>> list(
    ...  k_fold_cross_validation(range(6), 3)
    ... )  # doctest: +NORMALIZE_WHITESPACE
    [([4, 1, 0, 3], [2, 5]), ([2, 5, 0, 3], [4, 1]), ([2, 5, 4, 1], [0, 3])]
    """
    assert variant in config.SPLITTING_VARIANTS
    assert len(associations) >= k
    associations = list(associations)  # don't modify input with inplace shuffle
    r = random.Random(seed)
    r.shuffle(associations)
    part_len = len(associations) / k
    partitions = []
    for i in range(k):
        start_idx = int(i * part_len)
        end_idx = int((i + 1) * part_len)
        partitions.append(associations[start_idx:end_idx])
    for i in range(k):
        train = list(chain(*(partitions[:i] + partitions[i + 1:])))
        val = partitions[i]
        yield filter_node_pairs_split(train, val, variant)


def get_20_shuffled_pages_for_local_ground_truth_re_eval(verified_mappings):
    """get 20 shuffled pages.

    they were written into:
    './../eat/ver_sem_assocs/verify_semantic_associations_stimuli.txt'
    """
    re_test = list({v['stimulus'] for k, v in verified_mappings.items()})
    random.shuffle(re_test)
    re_test = re_test[:100]

    for _ in range(50):
        pick_20 = list(re_test)
        random.shuffle(pick_20)
        pick_20 = pick_20[:20]
        for s in pick_20:
            print(s)
        print()


def get_all_wikipedia_stimuli_for_triplerater(verified_mappings):
    # get all wikipedia stimuli for triplerater
    fn = './../eat/ver_sem_assocs/verify_semantic_associations_stimuli.txt'
    re_test_stimuli = open(fn).read()
    re_test_stimuli = [s.strip() for s in re_test_stimuli.split()]
    re_test_stimuli = set([s for s in re_test_stimuli if s])
    tr_test = list({
        v['stimulus_link']
        for k, v in verified_mappings.items()
        if v['stimulus'] in re_test_stimuli
    })
    for s in sorted(tr_test):
        print(s)


def main():
    import numpy as np
    import logging.config
    logging.basicConfig(level=logging.INFO)

    verified_mappings = get_verified_mappings()

    # get_dbpedia_pairs_from_mappings(verified_mappings)
    # sys.exit(0)

    # get_all_wikipedia_stimuli_for_triplerater(verified_mappings)
    # sys.exit(0)

    train, verify = split_mapping_training_test_set(verified_mappings)
    # pprint(verified_mappings)
    print("verified mappings {} ({} raw associations)".format(
        len(verified_mappings),
        sum([int(m['count']) for m in verified_mappings.values()]),
    ))
    print("used for training", len(train))
    print("used for eval", len(verify))
    # for v in verify.values():
    #     print(v)
    for split in [train, verify]:
        a = np.array([int(v['count'])/100 for v in split.values()])
        print('avg association strength:', a.mean(), 'stddev', a.std())

    sem_assocs = get_semantic_associations(None)
    if not path.isfile(config.GT_ASSOCIATIONS_FILENAME):
        write_semantic_associations(sem_assocs)
        print("created {}".format(config.GT_ASSOCIATIONS_FILENAME))
    assert get_semantic_associations(config.GT_ASSOCIATIONS_FILENAME) == \
        sem_assocs


if __name__ == '__main__':
    main()
