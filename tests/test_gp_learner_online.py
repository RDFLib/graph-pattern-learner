# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Online tests for gp_learner.

The following tests are depending on the endpoint and its loaded datasets.
In case of errors also check the used endpoint and if the tests make sense.
"""

import logging
from collections import OrderedDict

import SPARQLWrapper
import rdflib
from rdflib import URIRef
from rdflib import Variable

from config import SPARQL_ENDPOINT
from gp_learner import evaluate
from gp_learner import mutate_fix_var
from gp_learner import update_individuals
from gp_query import calibrate_query_timeout
from gp_query import query_time_hard_exceeded
from gp_query import query_time_soft_exceeded
from graph_pattern import GraphPattern
from graph_pattern import SOURCE_VAR
from graph_pattern import TARGET_VAR
from ground_truth_tools import get_semantic_associations
from ground_truth_tools import split_training_test_set
from gtp_scores import GTPScores

logger = logging.getLogger(__name__)

a = URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type')
dbp = rdflib.Namespace('http://dbpedia.org/resource/')
wikilink = URIRef('http://dbpedia.org/ontology/wikiPageWikiLink')

sparql = SPARQLWrapper.SPARQLWrapper(SPARQL_ENDPOINT)
try:
    timeout = max(5, calibrate_query_timeout(sparql))  # 5s for warmup
except IOError:
    from nose import SkipTest
    raise SkipTest(
        "Can't establish connection to SPARQL_ENDPOINT:\n    %s\n"
        "Skipping tests in\n    %s" % (SPARQL_ENDPOINT, __file__))
ground_truth_pairs = get_semantic_associations()
ground_truth_pairs, _ = split_training_test_set(ground_truth_pairs)
gtp_scores = GTPScores(ground_truth_pairs)


def test_evaluate():
    gp = GraphPattern((
        (SOURCE_VAR, wikilink, TARGET_VAR),
        (SOURCE_VAR, a, URIRef('http://dbpedia.org/ontology/PopulatedPlace')),
        (TARGET_VAR, a, URIRef('http://schema.org/Country'))
    ))
    res = evaluate(
        sparql, timeout, gtp_scores, gp)
    # (655, 0.4048, 0.4048, 0.0089, 7.5, 3, 3, 2, 0, 0.1936)
    # (remains, score, gain, f_measure, avg_reslens, gt_matches,
    #  patlen, patvars, timeout, qtime)
    update_individuals([gp], [res])
    fitness = gp.fitness.values
    matching_node_pairs = gp.matching_node_pairs
    gtp_precisions = gp.gtp_precisions
    gp.matching_node_pairs = matching_node_pairs
    logger.info(gp.matching_node_pairs)

    assert fitness.remains == len(ground_truth_pairs), 'remains wrong?'
    assert fitness.gt_matches == 3, "didn't match 3 gt pairs?"
    score = fitness.score
    assert 0 < score < 0.5, 'score not correct?'
    assert score == fitness.gain, 'score and gain should be the same here'
    assert 0 < fitness.f_measure < 0.1, \
        'f1 measure not correct?'
    assert fitness.patlen == 3, 'pattern should have 3 triples'
    assert fitness.patvars == 2, 'pattern should have 2 vars'
    if not query_time_soft_exceeded(fitness.qtime, timeout):
        assert 0 < fitness.avg_reslens < 10, \
            'avg match count should be ~7.5'
        assert fitness.timeout == 0, 'should not be a timeout'
    else:
        assert 0 < fitness.avg_reslens < 15, \
            'avg match count out of bounds for timeout'
        assert fitness.timeout > 0, 'should be a timeout'

    assert isinstance(gtp_precisions, OrderedDict)
    assert list(gtp_precisions) == matching_node_pairs
    logger.info(gtp_precisions)
    assert sum(gtp_precisions.values()) == fitness.gain, \
        'sum of precisions should be gain in this case'


def test_mutate_fix_var():
    # tests on a small subset
    ground_truth_pairs_ = [
        (dbp['Armour'], dbp['Knight']),
        (dbp['Barrel'], dbp['Wine']),
        (dbp['Barrister'], dbp['Law']),
        (dbp['Barrister'], dbp['Lawyer']),
        (dbp['Beak'], dbp['Bird']),
        (dbp['Beetroot'], dbp['Red']),
        (dbp['Belief'], dbp['Religion']),
        (dbp['Blanket'], dbp['Bed']),
        (dbp['Boot'], dbp['Shoe']),
        (dbp['Brine'], dbp['Salt']),
    ]
    v = Variable('v')
    gtp_scores_ = GTPScores(ground_truth_pairs_)
    gp = GraphPattern([
        (SOURCE_VAR, v, TARGET_VAR),
    ])
    tgps = mutate_fix_var(sparql, timeout, gtp_scores_, gp)
    assert tgps
    for tgp in tgps:
        logger.info(tgp.to_sparql_select_query())
        assert gp != tgp
        assert v not in tgp.vars_in_graph
    gp = GraphPattern([
        (SOURCE_VAR, v, TARGET_VAR),
        (SOURCE_VAR, a, Variable('source_type')),
        (TARGET_VAR, a, URIRef('http://schema.org/Country')),
    ])
    tgps = mutate_fix_var(sparql, timeout, gtp_scores_, gp, rand_var=v)
    assert tgps
    for tgp in tgps:
        logger.info(tgp.to_sparql_select_query())
        assert gp == tgp, 'should not have found any substitution'
    ground_truth_pairs_ = ((dbp['Berlin'], dbp['Germany']),)
    gtp_scores_ = GTPScores(ground_truth_pairs_)
    tgps = mutate_fix_var(sparql, timeout, gtp_scores_, gp)
    assert tgps
    for tgp in tgps:
        logger.info(tgp.to_sparql_select_query())
        assert gp != tgp, 'should have found a substitution'
        assert gp.vars_in_graph - tgp.vars_in_graph


def test_timeout_pattern():
    u = URIRef('http://dbpedia.org/resource/Template:Reflist')
    wpdisambig = URIRef('http://dbpedia.org/ontology/wikiPageDisambiguates')
    gp = GraphPattern([
        (SOURCE_VAR, Variable('v1'), u),
        (SOURCE_VAR, Variable('v5'), u),
        (TARGET_VAR, Variable('v0'), u),
        (TARGET_VAR, Variable('v3'), u),
        (TARGET_VAR, Variable('v6'), Variable('v2')),
        (Variable('v4'), wpdisambig, TARGET_VAR),
    ])
    res = evaluate(
        sparql, timeout, gtp_scores, gp)
    update_individuals([gp], [res])
    fitness = gp.fitness.values
    matching_node_pairs = gp.matching_node_pairs
    gp.matching_node_pairs = matching_node_pairs
    logger.info(gp.matching_node_pairs)
    assert query_time_soft_exceeded(fitness.qtime, timeout)
    assert fitness.score == 0
    if query_time_hard_exceeded(fitness.qtime, timeout):
        assert fitness.f_measure == 0
    else:
        assert fitness.f_measure > 0
