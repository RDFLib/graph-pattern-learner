#!/usr/bin/env python2.7
# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from copy import deepcopy
from itertools import combinations
from itertools import combinations_with_replacement
from itertools import permutations
from itertools import product
import sys

import networkx as nx
from rdflib import Variable
from scipy.special import binom
from scipy.misc import comb
from scoop.futures import map as parallel_map
import scoop
import scoop.futures
from splendid import chunker

from logging_config import logging
from graph_pattern import SOURCE_VAR
from graph_pattern import TARGET_VAR
from graph_pattern import GraphPattern
from graph_pattern import canonicalize
from graph_pattern import to_nx_graph

logger = logging.getLogger(__name__)
logger.info('init')


DEBUG = False
HOLE = sys.maxint  # placeholder for holes in partial patterns

# debug logging in this module is actually quite expensive (> 30 % of time). In
# case it's undesired the following removes that overhead.
if not DEBUG:
    # noinspection PyUnusedLocal
    def quick_skip_debug_log(*args, **kwds):
        pass
    logger.debug = quick_skip_debug_log


def numerical_patterns(
        length,
        loops=True,
        node_edge_joint=True,
        _partial_pattern=None,
        _pos=None,
        _var=1,
):
    """Numerical pattern generator.

    A pattern is a tuple of 3 tuples of variables, so for example the following
    is a pattern of length 2:
    ((?source, ?v3, ?target), (?target, ?v3, ?v4))

    For brevity, we can write the same as:
    'acb bcd' or numerical as '132 231'

    In the short version we could map ?source to 'a' or '1', ?target to 'b' or
    '2' and the other variables to the following letters / numbers.

    During generation we should take care that we don't generate a whole lot of
    unnecessary duplicates (so patterns that are obviously invalid or isomorphic
    to previous ones).

    A pattern is valid if:
    - its triples are sorted
        NO: 221 112 --> YES: 112 221
    - its triples are pairwise distinct
        NO: 112 112
    - its triples are pairwise connected
        NO: 123 456
        YES: 123 345
        YES: 123 132
    - the used variables don't skip a variable
        NO: 124 456 --> YES: 123 345
    - variables aren't unnecessary high
        NO: 124 334 --> YES: 123 443
        NO: 421 534 --> YES: 123 451
        YES: 312 411
    - it uses between 2 (source and target) and 2n + 1 vars (3 + 2 + 2 + ...)

    """
    if not _partial_pattern:
        _partial_pattern = [[HOLE, HOLE, HOLE] for _ in range(length)]
        _pos = (0, 0)

    i, j = _pos
    _partial_pattern = deepcopy(_partial_pattern)
    _partial_pattern[i][j] = _var

    if i >= 1 and _partial_pattern[i - 1] >= _partial_pattern[i]:
        # current triple must be larger than previous one for sorting and to
        # exclude multiple equivalent triples
        return


    # check if nodes and edges are disjoint
    if not node_edge_joint:
        flat_pp = [v for t in _partial_pattern for v in t]
        end = i*3 + j + 1  # end including last var
        nodes = set(flat_pp[0:end:3] + flat_pp[2:end:3])
        edges = set(flat_pp[1:end:3])
        if nodes & edges:
            logger.debug(
                'excluded node-edge-joined: %s', _partial_pattern[:i+1])
            return

    if j == 2:  # we just completed a triple
        # check for loops if necessary
        if not loops:
            s, _, o = _partial_pattern[i]
            if s == o:
                logger.debug('excluded loop: %s', _partial_pattern[:i+1])
                return

        if i >= 1:  # we're in a follow-up triple (excluding first)
            # check that it's connected
            s, p, o = _partial_pattern[i]
            for pt in _partial_pattern[:i]:
                # loop over previous triples and check if current is connected
                if s in pt or p in pt or o in pt:
                    # for p_only_connected it's possible to become
                    # n_connected again later:
                    # 123 145 627 685
                    #          ^    ^
                    break
            else:
                # we're not connected, early terminate this
                # This is safe as a later triple can't reconnect us anymore
                # without an isomorphic, lower enumeration that would've been
                # encountered before:
                # say we have
                #   abc xyz uvw
                # with xyz not being connected yet and uvw or any later part
                # connecting xyz back to abc. We can just use a breadth first
                # search from abc via those connecting triples and re-label all
                # encountered vars by breadth first search encountering. That
                # re-labeling is guaranteed to forward connect and it will
                # generate a smaller labelling than the current one.
                return

    if i == length - 1 and j == 2:
        # we're at the end of the pattern
        yield _partial_pattern
    else:
        # advance to next position
        j += 1
        if j > 2:
            j = 0
            i += 1

        flat_pp = [v for t in _partial_pattern for v in t]
        prev_vars = [v for v in flat_pp][:3*i + j]
        prev_max_var = max([v for v in prev_vars if v != HOLE])
        _star_var = 1
        # if i > 0:
        #     # doesn't seem to hold :(
        #     _star_var = _partial_pattern[i - 1][j]
        _end_var = min(
            prev_max_var + 1,  # can't skip a var
            # 2*length + 1,  # can't exceed max total number of vars (induced)
            3 + 2*i,  # vars in triple i can't exceed this, otherwise not sorted
        )
        for v in range(_star_var, _end_var + 1):
            for pattern in numerical_patterns(
                    length,
                    loops=loops,
                    node_edge_joint=node_edge_joint,
                    _partial_pattern=_partial_pattern,
                    _pos=(i, j),
                    _var=v
            ):
                yield pattern


def patterns(
        length,
        loops=True,
        node_edge_joint=True,
        p_only_connected=True,
        source_target_edges=True,
        exclude_isomorphic=True,
        count_candidates_only=False,
):
    """Takes a numerical pattern and generates actual patterns from it."""
    assert not count_candidates_only or not exclude_isomorphic, \
        'count_candidates_only cannot be used with isomorphism check'
    assert not source_target_edges or node_edge_joint, \
        'source_target_edges cannot be used without node_edge_joint'

    canonicalized_patterns = {}

    pid = -1
    for c, num_pat in enumerate(numerical_patterns(
            length,
            loops=loops,
            node_edge_joint=node_edge_joint,
    )):
        flat_num_pat = [v for t in num_pat for v in t]
        all_numbers = set(flat_num_pat)

        if not p_only_connected:
            # Numerical patterns are always connected, but they might be
            # p_only_connected (e.g., 123 425).
            # Check that the pattern isn't p_only_connected, meaning that it's
            # also connected by nodes (e.g., 123 325).
            # Note that in case of node_edge_joint 123 245 is also considered
            # p_only_connected.
            if not nx.is_connected(to_nx_graph(num_pat)):
                logger.debug('excluded %d: not node connected:\n%s', c, num_pat)
                continue

        if source_target_edges:
            all_numbers = sorted(all_numbers)
            numbers = all_numbers
        else:
            numbers = sorted(all_numbers - set(flat_num_pat[1::3]))
            all_numbers = sorted(all_numbers)

        if count_candidates_only:
            l = len(numbers)
            perms = l * (l-1)
            pid += perms
            # yield pid, None  # way slower, rather show progress from here:
            if c % 100000 == 0:
                logger.info(
                    'pattern id: %d, vars: %d, permutations: %d',
                    pid, l, perms
                )
            continue

        for s, t in permutations(numbers, 2):
            pid += 1
            # source and target are mapped to numbers s and t
            # re-enumerate the leftover numbers to close "holes"
            leftover_numbers = [n for n in all_numbers if n != s and n != t]
            var_map = {n: Variable('v%d' % i)
                       for i, n in enumerate(leftover_numbers)}
            var_map[s] = SOURCE_VAR
            var_map[t] = TARGET_VAR
            gp = GraphPattern(
                tuple([tuple([var_map[i] for i in trip]) for trip in num_pat]))

            # exclude patterns which are isomorphic to already generated ones
            if exclude_isomorphic:
                cgp = canonicalize(gp)
                if cgp in canonicalized_patterns:
                    igp = canonicalized_patterns[cgp]
                    igp_numpat, igp_s, igp_t, igp_gp = igp
                    logger.debug(
                        'excluded isomorphic %s with ?s=%d, ?t=%d:\n'
                        'isomorphic to %s with ?s=%d, ?t=%d:\n'
                        '%sand\n%s',
                        num_pat, s, t,
                        igp_numpat, igp_s, igp_t,
                        gp, igp_gp,
                    )
                    continue
                else:
                    canonicalized_patterns[cgp] = (num_pat, s, t, gp)
                    gp = cgp
            yield pid, gp
    yield pid + 1, None


def pattern_generator(
        length,
        loops=True,
        node_edge_joint=True,
        p_only_connected=True,
        source_target_edges=True,
        exclude_isomorphic=True,
        count_candidates_only=False,
):
    assert not source_target_edges or node_edge_joint, \
        'source_target_edges cannot be used without node_edge_joint'
    canonicalized_patterns = {}

    if node_edge_joint:
        # To be connected there are max 3 + 2 + 2 + 2 + ... vars for triples.
        # The first can be 3 different ones (including ?source and ?target, then
        # in each of the following triples at least one var has to be an old one
        possible_vars = [Variable('v%d' % i) for i in range((2 * length) - 1)]
        possible_nodes = possible_vars + [SOURCE_VAR, TARGET_VAR]
        if source_target_edges:
            possible_edges = possible_nodes
        else:
            possible_edges = possible_vars
    else:
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
    if count_candidates_only:
        yield (n_patterns, None)
        return

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
        edges = sorted(gp.edges - {SOURCE_VAR, TARGET_VAR})
        vars_ = sorted(gp.vars_in_graph - {SOURCE_VAR, TARGET_VAR})

        # check there are no skipped variables (nodes or edges)
        # noinspection PyUnboundLocalVariable
        if (
                (node_edge_joint and vars_ != possible_vars[:len(vars_)]) or
                (not node_edge_joint and (
                    nodes != possible_var_nodes[:len(nodes)] or
                    edges != possible_edges[:len(edges)]
                ))
        ):
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
        if not gp.is_connected(via_edges=p_only_connected):
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
    # len | pcon | nej | all          | candidates (all)  | candidates (all)  |
    #     |      |     | (canonical)  | (old method)      | (numerical)       |
    # ----+------+-----+--------------+-------------------+-------------------+
    #   1 |    8 |  12 |           12 |                27 |                12 |
    #   2 |  146 | 469 |          693 |              7750 |              1314 |
    #   3 |      |     |        47478 |           6666891 |            151534 |
    #   4 |      |     |              |       11671285626 |          20884300 |
    #   5 |      |     |              |    34549552710596 |        3461471628 |

    # len | typical     | candidates     | candidates  |
    #     | (canonical) | (old method)   | (numerical) |
    # ----+-------------+----------------+-------------+
    #   1 |           2 |              4 |           2 |
    #   2 |          28 |            153 |          54 |
    #   3 |         486 |          17296 |        1614 |
    #   4 |       10374 |        3921225 |       59654 |
    #   5 |             |     1488847536 |     2707960 |

    # typical above means none of (loops, nej, pcon, source_target_edges)

    length = 5
    canonical = True

    _patterns = set()
    n = 0
    i = 0

    pg = patterns(
        length,
        loops=False,
        node_edge_joint=False,
        p_only_connected=False,
        source_target_edges=False,
        exclude_isomorphic=canonical and not scoop.IS_RUNNING,
        count_candidates_only=False,
    )

    if canonical and scoop.IS_RUNNING:
        # Graph pattern isomorphism checking is what takes by far the longest.
        # run canonicalization in parallel
        # chunks used for efficiency and to hinder parallel_map from trying to
        # eat up all candidates first
        for chunk in chunker(pg, 10000):
            cgps = parallel_map(
                lambda res: (res[0], canonicalize(res[1]) if res[1] else None),
                chunk
            )
            for i, pattern in cgps:
                if pattern not in _patterns:
                    print('%d: Pattern id %d: %s' % (n, i, pattern))
                    _patterns.add(pattern)
                    n += 1
    else:
        # run potential canonicalization inline
        for n, (i, pattern) in enumerate(pg):
            print('%d: Pattern id %d: %s' % (n, i, pattern))
            _patterns.add(pattern)
    # last res of pg is (i, None)
    _patterns.remove(None)
    print('Number of pattern candidates: %d' % i)
    print('Number of patterns: %d' % n)

    # testing flipped edges (only works if we're working with canonicals)
    if canonical:
        for gp in _patterns:
            for i in range(length):
                mod_gp = gp.flip_edge(i)
                # can happen that flipped edge was there already
                if len(mod_gp) == length:
                    cmod_pg = canonicalize(mod_gp)
                    assert cmod_pg in _patterns, \
                        'gp: %smod_gp: %scanon: %s_patterns: %r...' % (
                            gp, mod_gp, cmod_pg, list(_patterns)[:20]
                        )


if __name__ == '__main__':
    main()
