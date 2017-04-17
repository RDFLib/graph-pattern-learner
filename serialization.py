#!/usr/bin/env python2.7
# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
from collections import Counter
from collections import OrderedDict
from datetime import datetime
import json
from glob import glob
import gzip
import logging
import os
from os import path
import shelve
from time import sleep

import deap
import deap.base
import deap.tools
import numpy as np
from splendid import run_once

import config
from graph_pattern import GraphPattern
from gtp_scores import GTPScores
from utils import decurify

logger = logging.getLogger(__name__)


def format_graph_pattern(gp, matching_node_pairs=10):
    assert isinstance(gp, GraphPattern)
    res = ['Fitness: %s\n%s\n%s' % (
        gp.fitness.format_fitness(),
        gp.fitness.description,
        gp
    )]
    if matching_node_pairs > 0 and gp.matching_node_pairs:
        res.append("matching node pairs:")
        for s, t in gp.matching_node_pairs[:matching_node_pairs]:
            res.append('  %s %s' % (gp.curify(s), gp.curify(t)))
        if len(gp.matching_node_pairs) > matching_node_pairs:
            res.append('  ...')
    res.append('\n')
    return '\n'.join(res)


def print_graph_pattern(gp, print_matching_node_pairs=10):
    print(format_graph_pattern(gp, print_matching_node_pairs))


def print_population(run, ngen, population, n=10):
    sleep(.5)  # syncing of stderr and stdout never seems to work otherwise
    print("\n\n\nrun %d, population generation %d:\n" % (run, ngen))
    c = Counter(population)
    top_n = list(deap.tools.selBest(c.keys(), n))
    for gp in top_n:
        assert isinstance(gp, GraphPattern)
        print('GraphPattern %d times in population:' % c[gp])
        print_graph_pattern(gp)
    return c


def save_population(run, ngen, top_gps, gtp_scores):
    patterns_found_in_run = [(gp, run) for gp in top_gps]
    file_prefix = 'top_graph_patterns_run_%02d_gen_%02d' % (run, ngen)
    file_path = save_results(
        patterns_found_in_run,
        gtp_scores=gtp_scores,
        run=run,
        ngen=ngen,
        file_prefix=file_prefix)
    set_symlink(file_path, config.SYMLINK_CURRENT_RES_RUN_GEN)


def set_symlink(file_path, symlink_name):
    symlink_path = path.join(config.RESDIR, symlink_name)
    if path.islink(symlink_path):
        os.remove(symlink_path)
    if hasattr(os, 'symlink'):
        os.symlink(path.basename(file_path), symlink_path)


def remove_old_result_files():
    for fn in glob(path.join(config.RESDIR, 'top_graph_patterns_*.json.gz')):
        os.remove(fn)
    for fn in glob(path.join(config.RESDIR,
                             config.RES_RUN_PREFIX + '_*.json.gz')):
        os.remove(fn)
    for sln in [config.SYMLINK_CURRENT_RES_RUN_GEN,
                config.SYMLINK_CURRENT_RES_RUN]:
        symlink_path = path.join(config.RESDIR, sln)
        if path.islink(symlink_path):
            os.remove(symlink_path)


def save_results(
        patterns_found_in_run,
        coverage_counts=None,
        gtp_scores=None,
        overall_gtp_scores=None,
        run=None,
        ngen=None,
        file_path=None,
        file_prefix='results',
        **kwds
):
    now = datetime.now()
    if not file_path:
        timestamp = now.strftime('%Y-%m-%dT%H-%M-%S')
        file_path = path.join(config.RESDIR,
                              file_prefix + '_%s.json.gz' % timestamp)
    res = {
        'timestamp': str(now),

        # sadly the following transformations can't be realized as a JSONEncoder
        # subclass as its default method is only ever called if the objects are
        # not instances of basic objects. As GraphPattern and Identifiers are
        # subclasses of tuple / string, the default method would never be called
        # and GraphPattern just be serialized as tuple and Identifiers as str.
        # Also the default method is never called at all for dict keys...
        # So you see, the standard json lib is very "customizable"... NOT! GRRR
        'patterns': [
            {'graph_pattern': gp.to_dict(), 'found_in_run': r}
            for gp, r in patterns_found_in_run
        ],
    }

    # JSON dict keys can't be compound objects, but would end up as strings.
    # That would cause nasty double parsing if ever read, so the following
    # just transforms it into lists with a pair as first element, meaning
    # Identifiers will not be smushed together by repr into a 'key' str.
    if coverage_counts is not None:
        res['coverage_counts'] = [
            ((s.n3(), t.n3()), c)
            for (s, t), c in sorted(coverage_counts.items())
        ]

    if gtp_scores is not None:
        gtps = gtp_scores.ground_truth_pairs
        res['ground_truth_pairs'] = [
            (s.n3(), t.n3())
            for s, t in gtps
        ]

        # this one is used by the visualisation to show accumulated precisions
        # it should only contain those of the current patterns
        coverage_max_precision = gtp_scores.gtp_max_precisions
        res['coverage_max_precision'] = [
            ((s.n3(), t.n3()), p)
            for (s, t), p in sorted(coverage_max_precision.items())
        ]

    if overall_gtp_scores is not None:
        overall_coverage_max_precision = overall_gtp_scores.gtp_max_precisions
        res['overall_coverage_max_precision'] = [
            ((s.n3(), t.n3()), p)
            for (s, t), p in sorted(overall_coverage_max_precision.items())
        ]

    if run is not None:
        res['run_number'] = run
    if ngen is not None:
        res['generation_number'] = ngen

    if kwds:
        res.update(**kwds)
    try:
        os.makedirs(path.dirname(file_path))
    except OSError:
        pass
    with gzip.open(file_path, 'w') as f:
        json.dump(res, f, indent=2)
    return file_path


def find_last_result():
    # will only work the next 985 years ;-/
    result_file_names = glob(path.join(config.RESDIR, 'results_2*.json.gz'))
    if result_file_names:
        return sorted(result_file_names)[-1]
    else:
        return None


def find_run_result(run):
    fn = glob(path.join(config.RESDIR, config.RES_RUN_PREFIX + '_%02d_*' % run))
    if fn:
        return sorted(fn)[-1]
    else:
        return None


def load_results(fn):
    logger.info('loading results from: %s', fn)
    with gzip.open(fn) as f:
        res = json.load(f)
    result_patterns = [
        (GraphPattern.from_dict(pattern_run['graph_pattern']),
         pattern_run['found_in_run'])
        for pattern_run in res['patterns']
    ]
    coverage_counts = Counter({
        (decurify(s), decurify(t)): c
        for (s, t), c in res.get('coverage_counts', [])
    })
    gtp_scores = None
    gtps = [tuple(gtp) for gtp in res.get('ground_truth_pairs')]
    if gtps:
        coverage_max_precision = res.get('overall_coverage_max_precision', [])
        if not coverage_max_precision:
            # final result file for example
            coverage_max_precision = res.get('coverage_max_precision', [])
        gtp_scores = GTPScores(gtps)
        gtp_scores.gtp_max_precisions = OrderedDict([
            ((decurify(s), decurify(t)), mp)
            for (s, t), mp in coverage_max_precision
        ])
    logger.info('loaded %d result patterns', len(result_patterns))
    return result_patterns, coverage_counts, gtp_scores


def print_results(
        result_patterns, coverage_counts, gtp_scores,
        n=None,
        edge_only_connected_patterns=True,
):
    coverage_max_precision = gtp_scores.gtp_max_precisions
    if n is None or n > 0:
        print('\n\n\nGraph pattern learner raw result patterns:')
        if n and n < len(result_patterns):
            print('(only printing top %d out of %d found patterns)' % (
                n, len(result_patterns)))
        print()
        for gp, run in result_patterns[:n]:
            print('Pattern from run %d:' % run)
            print_graph_pattern(gp)

    if edge_only_connected_patterns:
        @run_once
        def lazy_print_header():
            print(
                '\n\n\nThe following edge only connected or mixed node and edge'
                ' vars patterns made it into raw result patterns:\n'
            )
        for gp, run in result_patterns:
            if gp.is_edge_connected_only():
                lazy_print_header()
                print('edge connected only pattern:')
                print_graph_pattern(gp, 0)
            if gp.mixed_node_edge_vars():
                lazy_print_header()
                print('edges and nodes joint in pattern:')
                print_graph_pattern(gp, 0)

    print('\n\n\nCoverage stats:\nPatterns, Max Precision, Stimulus, Response')
    for gtp, count in coverage_counts.most_common():
        print(
            '%3d %.3f %s %s' % (
                count, coverage_max_precision[gtp], gtp[0].n3(), gtp[1].n3()
            )
        )

    print('\nMax Precision Histogram:')
    print('      %  :   h,   H')
    hist, bin_edges = np.histogram(coverage_max_precision.values())
    sum_h = 0
    for h, edge in zip(hist, bin_edges):
        sum_h += h
        print('  >= %.2f: %3d, %3d' % (edge, h, sum_h))

    max_score = gtp_scores.score
    print('\n\noverall score (precision sum on training set): %.3f' % max_score)
    print('training set length: %d' % len(coverage_counts))

    print(
        'expected recall with good rank (precision sum / len(training set)): '
        '%.3f' % (max_score / len(coverage_counts))
    )
    cov_max_prec_gt_0 = len(
        [mp for gtp, mp in coverage_max_precision.items() if mp > 0]
    )
    print(
        'expected recall without rank limit: %.3f\n\n' % (
            cov_max_prec_gt_0 / len(coverage_counts))
    )
