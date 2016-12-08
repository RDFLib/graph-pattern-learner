#!/usr/bin/env python2.7
# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from itertools import combinations
from itertools import combinations_with_replacement
from itertools import permutations
from itertools import product

from rdflib import Variable
from scipy.special import binom

from logging_config import logging
from graph_pattern import SOURCE_VAR
from graph_pattern import TARGET_VAR
from graph_pattern import GraphPattern
from graph_pattern import canonicalize

logger = logging.getLogger(__name__)
logger.info('init')


DEBUG = False

# debug logging in this module is actually quite expensive (> 30 % of time). In
# case it's undesired the following removes that overhead.
if not DEBUG:
    # noinspection PyUnusedLocal
    def quick_skip_debug_log(*args, **kwds):
        pass
    logger.debug = quick_skip_debug_log


def pattern_generator(
        length,
        loops=True,
        node_edge_joint=True,
        p_connected=True,
        exclude_isomorphic=True,
):
    canonicalized_patterns = {}

    # To be connected there are max 3 + 2 + 2 + 2 + ... vars for the triples.
    # The first can be 3 different ones (including ?source and ?target, then
    # in each of the following triples at least one var has to be an old one
    possible_vars = [Variable('v%d' % i) for i in range((2 * length) - 1)]
    possible_vars += [SOURCE_VAR, TARGET_VAR]

    possible_triples = [
        (s, p, o)
        for s in possible_vars
        for p in possible_vars
        for o in possible_vars
    ]

    n_patterns = binom(len(possible_triples), length)
    logger.info(
        'generating %d possible patterns of length %d', n_patterns, length)

    i = 0
    pid = 0
    for pid, pattern in enumerate(combinations(possible_triples, length)):
        gp = GraphPattern(pattern)

        # check that source and target are in gp:
        if not gp.complete():
            logger.debug(
                'excluded %d: source or target missing: %s', pid, gp)
            continue
        vars_ = sorted(gp.vars_in_graph - {SOURCE_VAR, TARGET_VAR})

        # check there are no skipped nodes, e.g., link to n2 picked but no n1
        if vars_ != possible_vars[:len(vars_)]:
            logger.debug('excluded %d: skipped var: %s', pid, gp)
            continue

        # check if nodes and edges are disjoint
        if not node_edge_joint and (gp.nodes & gp.edges):
            logger.debug('excluded %d: node-edge-joined: %s', pid, gp)
            continue

        # check for loops if necessary
        if not loops and any([s == o for s, p, o in gp]):
            logger.debug('excluded %d: loop: %s', pid, gp)
            continue

        # check that the pattern is connected
        if not gp.is_connected(via_edges=p_connected):
            logger.debug('excluded %d: not connected:\n%s', pid, gp)
            continue

        # exclude patterns which are isomorphic to already generated ones
        if exclude_isomorphic:
            cgp = canonicalize(gp)
            if cgp in canonicalized_patterns:
                logger.debug(
                    'excluded %d: isomorphic to %d:\n%sand\n%s',
                    pid,
                    canonicalized_patterns[cgp][0],
                    gp,
                    canonicalized_patterns[cgp][1]
                )
                continue
            else:
                canonicalized_patterns[cgp] = (pid, gp)
                gp = cgp
        i += 1
        logger.debug('generated pattern %d: %s', pid, gp)
        yield pid, gp
    assert pid + 1 == n_patterns
    logger.info(
        'found %d differing patterns out of %d possible of length %d',
        i, n_patterns, length
    )
    yield (n_patterns, None)


def main():
    length = 3
    # 3: 47478 (pcon, nej) of 6666891
    # 4:
    # 5:

    gen_patterns = []
    for n, (i, pattern) in enumerate(pattern_generator(length)):
        print('%d: Pattern id %d: %s' % (n, i, pattern))
        gen_patterns.append((i, pattern))
    patterns = set(gp for pid, gp in gen_patterns[:-1])

    # testing flipped edges
    for gp in patterns:
        for i in range(length):
            mod_gp = gp.flip_edge(i)
            # can happen that flipped edge was there already
            if len(mod_gp) == length:
                assert canonicalize(mod_gp) in patterns


if __name__ == '__main__':
    main()
