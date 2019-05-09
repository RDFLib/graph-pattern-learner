# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import logging
import random

import rdflib
from rdflib import Literal
from rdflib import URIRef
from rdflib import Variable
from scipy.stats import binom

from gp_learner import mutate_increase_dist
from gp_learner import mutate_merge_var
from gp_learner import mutate_simplify_pattern
from graph_pattern import GraphPattern
from graph_pattern import SOURCE_VAR
from graph_pattern import TARGET_VAR
from ground_truth_tools import get_semantic_associations
from ground_truth_tools import split_training_test_set
from gtp_scores import GTPScores

logger = logging.getLogger(__name__)

dbp = rdflib.Namespace('http://dbpedia.org/resource/')
wikilink = URIRef('http://dbpedia.org/ontology/wikiPageWikiLink')

ground_truth_pairs = get_semantic_associations()
ground_truth_pairs, _ = split_training_test_set(ground_truth_pairs)
gtp_scores = GTPScores(ground_truth_pairs)


def test_mutate_increase_dist():
    gp = GraphPattern([(SOURCE_VAR, wikilink, TARGET_VAR)])
    res = mutate_increase_dist(gp)
    assert gp != res
    assert gp.diameter() + 1 == res.diameter()
    assert gp.vars_in_graph == {SOURCE_VAR, TARGET_VAR}


def test_mutate_merge_var():
    p = Variable('p')
    q = Variable('q')
    gp = GraphPattern([
        (SOURCE_VAR, p, TARGET_VAR)
    ])
    res = mutate_merge_var(gp, 0)
    assert res == gp
    res = mutate_merge_var(gp, 1)
    assert res[0][1] in {SOURCE_VAR, TARGET_VAR}

    gp2 = gp + [(SOURCE_VAR, q, TARGET_VAR)]
    res = mutate_merge_var(gp2, 0)
    assert len(res) == 1, "?q must have become ?p or vice versa: %s" % len(res)
    assert res[0][1] in {p, q}
    a, b = False, False
    for i in range(100):
        res = mutate_merge_var(gp2, 1)
        if len(res) == 1:
            assert res[0][1] in {p, q}
            a = True
        else:
            # one of the edge vars must have become ?s or ?t
            assert {res[0][1], res[1][1]} & {SOURCE_VAR, TARGET_VAR}
            assert {res[0][1], res[1][1]} - {SOURCE_VAR, TARGET_VAR}
            b = True
        if a and b:
            break
    else:
        assert False, "merge never reached one of two cases: %s %s" % (a, b)

    gp2 = gp + [(q, p, TARGET_VAR)]
    a, b = False, False
    for i in range(100):
        res = mutate_merge_var(gp2, 0)
        if len(res) == 1:
            # q must have become ?source
            assert res == gp
            a = True
        else:
            # q became ?target
            assert res == gp + [(TARGET_VAR, p, TARGET_VAR)]
            b = True
        if a and b:
            break
    else:
        assert False, "merge never reached one of two cases: %s %s" % (a, b)

    cases = [False] * 4
    for i in range(100):
        res = mutate_merge_var(gp2, 1)
        if len(res) == 1:
            # q must have become ?source
            assert res == gp
            cases[0] = True
        else:
            # ?q became ?target or ?p, or ?p one of {?q, ?source, ?target}
            if res == gp + [(TARGET_VAR, p, TARGET_VAR)]:
                cases[1] = True
            elif res == gp + [(p, p, TARGET_VAR)]:
                cases[2] = True
            else:
                assert res[0][1] in {q, SOURCE_VAR, TARGET_VAR}
                cases[3] = True
        if all(cases):
            break
    else:
        assert False, "merge never reached one of the cases: %s" % cases


def test_simplify_pattern():
    gp = GraphPattern([(SOURCE_VAR, wikilink, TARGET_VAR)])
    res = mutate_simplify_pattern(gp)
    assert gp == res, 'should not simplify simple pattern'

    # test parallel single var edges
    gp_bloated = gp + [
        (SOURCE_VAR, Variable('v1'), TARGET_VAR),
    ]
    res = mutate_simplify_pattern(gp_bloated)
    assert res == gp, 'not simplified:\n%s' % (res,)
    gp_bloated += [
        (SOURCE_VAR, Variable('v2'), TARGET_VAR),
    ]
    res = mutate_simplify_pattern(gp_bloated)
    assert res == gp, 'not simplified:\n%s' % (res,)

    # test edges between fixed nodes
    gp += [
        (SOURCE_VAR, wikilink, dbp['City']),
        (TARGET_VAR, wikilink, dbp['Country']),
    ]
    gp_bloated = gp + [
        (dbp['City'], wikilink, dbp['Country']),
        (dbp['Country'], Variable('v2'), dbp['City']),
    ]
    res = mutate_simplify_pattern(gp_bloated)
    assert res == gp, 'not simplified:\n%s' % (res,)

    # test unrestricting leaves:
    gp_bloated = gp + [
        (SOURCE_VAR, Variable('v3'), Variable('v4')),
    ]
    res = mutate_simplify_pattern(gp_bloated)
    assert res == gp, 'not simplified:\n%s' % (res,)
    gp_bloated = gp + [
        (SOURCE_VAR, Variable('v3'), Variable('v4')),
        (Variable('v5'), Variable('v6'), Variable('v4')),
    ]
    res = mutate_simplify_pattern(gp_bloated)
    assert res == gp, 'not simplified:\n%s' % (res,)
    gp_bloated = gp + [
        (SOURCE_VAR, Variable('v3'), Variable('v4')),
        (Variable('v5'), Variable('v6'), Variable('v4')),
        (Variable('v4'), Variable('v7'), Variable('v8')),
    ]
    res = mutate_simplify_pattern(gp_bloated)
    assert res == gp, 'not simplified:\n%s' % (res,)

    # test leaves behind fixed nodes
    gp += [
        (SOURCE_VAR, wikilink, Variable('v4')),
    ]
    gp_bloated = gp + [
        (Variable('v5'), wikilink, dbp['Country']),
        (Variable('v5'), Variable('v6'), Variable('v7')),
    ]
    res = mutate_simplify_pattern(gp_bloated)
    assert res == gp, 'not simplified:\n%s' % (res,)

    # counter example of an advanced but restricting pattern:
    gp += [
        (SOURCE_VAR, Variable('v3'), Variable('v4')),
        (Variable('v5'), Variable('v6'), Variable('v4')),
        (Variable('v4'), Variable('v7'), Variable('v8')),
        (TARGET_VAR, Variable('v3'), SOURCE_VAR),
        (dbp['City'], Variable('v6'), dbp['Country']),
        (dbp['Country'], Variable('v8'), dbp['City']),
    ]
    res = mutate_simplify_pattern(gp)
    assert res == gp, 'was simplified (bad):\n%s' % (res,)

    # test atomic patterns:
    gp = GraphPattern([
        (SOURCE_VAR, Variable('v1'), Variable('v2'))
    ])
    res = mutate_simplify_pattern(gp)
    assert res == gp, 'was simplified (bad):\n%s' % (res,)
    gp = GraphPattern([
        (SOURCE_VAR, Variable('v1'), Variable('v2')),
        (SOURCE_VAR, Variable('v3'), Variable('v4')),
    ])
    res = mutate_simplify_pattern(gp)
    assert res == gp, 'was simplified (bad):\n%s' % (res,)

    # test edge var connections
    gp = GraphPattern([
        (SOURCE_VAR, Variable('p'), Variable('v1')),
        (TARGET_VAR, Variable('p'), Variable('v2')),
    ])
    res = mutate_simplify_pattern(gp)
    assert res == gp, 'was simplified (bad):\n%s\nto\n%s' % (gp, res)
    gp2 = gp + [
        (Variable('v1'), Variable('v3'), Variable('v4')),
    ]
    res = mutate_simplify_pattern(gp2)
    assert res == gp, 'not simplified:\n%s\nto\n%s' % (gp2, res)
    gp = GraphPattern([
        (SOURCE_VAR, Variable('p'), Variable('v1')),
        (Variable('p'), Variable('v2'), TARGET_VAR),
    ])
    res = mutate_simplify_pattern(gp)
    assert res == gp, 'was simplified (bad):\n%s\nto\n%s' % (gp, res)
    gp2 = gp + [
        (Variable('p'), Variable('v3'), TARGET_VAR),
    ]
    res = mutate_simplify_pattern(gp2)
    assert res == gp, 'not simplified:\n%s\nto\n%s' % (gp2, res)

    # make sure that we keep literals
    gp = GraphPattern([
        (SOURCE_VAR, Variable('p'), Literal('foo')),
        (SOURCE_VAR, wikilink, Literal('bar')),
        (SOURCE_VAR, wikilink, TARGET_VAR),
        (TARGET_VAR, Variable('q'), Literal('bla')),
        (SOURCE_VAR, wikilink, Literal('blu')),
        (SOURCE_VAR, Variable('r'), Literal('foobar')),
        (TARGET_VAR, Variable('r'), Literal('foobar')),

    ])
    res = mutate_simplify_pattern(gp)
    assert res == gp, 'was simplified (bad):\n%s\nto\n%s' % (gp, res)


def test_remaining_gain_sample_gtps():
    n = len(ground_truth_pairs)
    gtps = sorted(gtp_scores.remaining_gain_sample_gtps(max_n=n))
    assert len(gtps) == n
    # if we draw everything the results should always be everything
    assert gtps == sorted(gtp_scores.remaining_gain_sample_gtps(max_n=n))
    # if we don't draw everything it's quite unlikely we get the same result
    gtps = gtp_scores.remaining_gain_sample_gtps(max_n=5)
    assert len(gtps) == 5
    assert gtps != gtp_scores.remaining_gain_sample_gtps(max_n=5)

    # make sure we never get items that are fully covered already
    gtp_scores.gtp_max_precisions[ground_truth_pairs[0]] = 1
    c = Counter()
    k = 100
    n = 128
    for i in range(k):
        c.update(gtp_scores.remaining_gain_sample_gtps(max_n=n))
    assert ground_truth_pairs[0] not in c
    assert sum(c.values()) == k * n
    # count how many aren't in gtps
    c_not_in = 0
    for gtp in ground_truth_pairs[1:]:
        if gtp not in c:
            c_not_in += 1
    assert c_not_in < 2, \
        "it's very unlikely that 2 gtps weren't in our %d samples, " \
        "but %d are not" % (k, c_not_in)

    # near end simulation
    gtpe_scores = gtp_scores.copy_reset()
    # set all scores to 1 --> remaining gains to 0
    gtpe_scores.gtp_max_precisions = gtpe_scores.get_remaining_gains()
    high_prob, low_prob = random.sample(gtpe_scores.ground_truth_pairs, 2)
    # high and low prob refer to the remaining gains and the expected probs to
    # be selected by remaining gain samples...
    gtpe_scores.gtp_max_precisions[high_prob] = 0.1
    gtpe_scores.gtp_max_precisions[low_prob] = 0.9
    assert gtpe_scores.remaining_gain == 1
    c = Counter()
    for i in range(100):
        c.update(gtpe_scores.remaining_gain_sample_gtps(max_n=1))
    assert len(c) == 2
    assert sum(c.values()) == 100
    assert (binom.pmf(c[high_prob], 100, .9) > 0.001 and
            binom.pmf(c[low_prob], 100, .1) > 0.001), \
        'expected that high_prob item is drawn with a 9:1 chance, but got:\n' \
        'high: %d, low: %d' % (c[high_prob], c[low_prob])


def test_gtp_scores():
    assert gtp_scores - gtp_scores == 0
