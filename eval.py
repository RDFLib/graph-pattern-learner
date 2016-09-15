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


# FIXME: properties as subject / object!

def pattern_generator(length, loops=True, exclude_isomorphic=True):
    canonicalized_patterns = {}
    possible_var_nodes = [Variable('n%d' % i) for i in range(length - 1)]
    possible_nodes = possible_var_nodes + [SOURCE_VAR, TARGET_VAR]
    possible_edges = [Variable('e%d' % i) for i in range(length)]

    possible_triples = [
        (s, p, o)
        for s in possible_nodes
        for p in possible_edges
        for o in possible_nodes
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
        nodes = sorted(gp.nodes - {SOURCE_VAR, TARGET_VAR})
        edges = sorted(gp.edges)

        # check there are no skipped nodes, e.g., link to n2 picked but no n1
        if nodes != possible_var_nodes[:len(nodes)]:
            logger.debug('excluded %d: skipped node: %s', pid, gp)
            continue
        if edges != possible_edges[:len(edges)]:
            logger.debug('excluded %d: skipped edge: %s', pid, gp)
            continue

        # check for loops if necessary
        if not loops and any([s == o for s, p, o in gp]):
            logger.debug('excluded %d: loop: %s', pid, gp)
            continue

        # check that the pattern is connected
        if not gp.is_connected():
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
    length = 5
    # 3: 702 of 17296
    # 4: 16473 of 3921225
    # 5:  of 1488847536

    gen_patterns = list(pattern_generator(length))
    for n, (i, pattern) in enumerate(gen_patterns):
        print('%d: Pattern id %d: %s' % (n, i, pattern))
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
