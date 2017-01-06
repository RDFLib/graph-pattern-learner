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

from rdflib import Variable
from scipy.special import binom
from scipy.misc import comb

from logging_config import logging
from graph_pattern import SOURCE_VAR
from graph_pattern import TARGET_VAR
from graph_pattern import GraphPattern
from graph_pattern import canonicalize

logger = logging.getLogger(__name__)
logger.info('init')


DEBUG = True
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
        p_connected=True,
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
                if s in pt or o in pt or (p_connected and p in pt):
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
                    p_connected=p_connected,
                    _partial_pattern=_partial_pattern,
                    _pos=(i, j),
                    _var=v
            ):
                yield pattern


def patterns(
        length,
        loops=True,
        node_edge_joint=True,
        p_connected=True,
        exclude_isomorphic=True,
        count_candidates_only=False,
):
    """Takes a numerical pattern and generates actual patterns from it."""
    assert not count_candidates_only or not exclude_isomorphic, \
        'count_candidates_only cannot be used with isomorphism check'

    canonicalized_patterns = {}

    pid = -1
    for c, num_pat in enumerate(numerical_patterns(
            length,
            loops=loops,
            node_edge_joint=node_edge_joint,
            p_connected=p_connected,
    )):
        numbers = sorted(set([v for t in num_pat for v in t]))
        # var_map = {i: '?v%d' % i for i in numbers}
        # pattern = GraphPattern(
        #     tuple([tuple([var_map[i] for i in t]) for t in numerical_repr]))
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
            leftover_numbers = [n for n in numbers if n != s and n != t]
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
    length = 1
    canonical = True
    # len | pcon | nej | pcon, nej    | candidates     | candidates  |
    #     |      |     | (canonical)  | (old method)   | (numerical) |
    # ----+------+-----+--------------+----------------+-------------+
    #   1 |    8 |  12 |           12 |             27 |          12 |
    #   2 |  146 | 469 |          693 |           7750 |        1314 |
    #   3 |      |     |        47478 |        6666891 |      151534 |
    #   4 |      |     |              |    11671285626 |    20884300 |
    #   5 |      |     |              | 34549552710596 |  3461471628 |

    gen_patterns = []
    n = 0
    i = 0
    for n, (i, pattern) in enumerate(patterns(
            length,
            loops=False,
            node_edge_joint=False,
            p_connected=False,
            exclude_isomorphic=canonical,
            count_candidates_only=False,
    )):
        print('%d: Pattern id %d: %s' % (n, i, pattern))
        gen_patterns.append((i, pattern))
    print('Number of pattern candidates: %d' % i)
    print('Number of patterns: %d' % n)
    _patterns = set(gp for pid, gp in gen_patterns[:-1])

    # testing flipped edges (only works if we're working with canonicals)
    if canonical:
        for gp in _patterns:
            for i in range(length):
                mod_gp = gp.flip_edge(i)
                # can happen that flipped edge was there already
                if len(mod_gp) == length:
                    cmod_pg = canonicalize(mod_gp)
                    assert cmod_pg in _patterns, \
                        'mod_gp: %r\ncanon: %r\n_patterns: %r' % (
                            mod_gp, cmod_pg, _patterns
                        )


if __name__ == '__main__':
    main()
