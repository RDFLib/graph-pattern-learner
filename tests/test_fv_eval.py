# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""test_mutate_fix_var und test_evaluate einmal davor und 
einmal über die results aus mutate_fix_var
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
from os import getenv

logger = logging.getLogger(__name__)

dbp = rdflib.Namespace('http://dbpedia.org/resource/')


v = Variable('v')

gp = GraphPattern([
        (SOURCE_VAR, v, TARGET_VAR),
    ])

ground_truth_pairs_ = [
    (dbp['Berlin'],dbp['Germany']),
    (dbp['Hamburg'],dbp['Germany']),
    (dbp['Kaiserslautern'],dbp['Germany']),
    (dbp['Wien'],dbp['Austria']),
    (dbp['Insbruck'],dbp['Austria']),
    (dbp['Salzburg'],dbp['Austria']),
    (dbp['Paris'],dbp['France']),
    (dbp['Lyon'],dbp['France']),
    (dbp['Amsterdam'],dbp['Netherlands']),
    (dbp['Brussels'],dbp['Belgium']),
    (dbp['Washington'],dbp['United_States']),
    (dbp['Madrid'],dbp['Spain']),
    (dbp['Prague'],dbp['Czech_Republic']),
    (dbp['Bern'],dbp['Switzerland']),
]

gtp_scores_ = GTPScores(ground_truth_pairs_)

sparql = SPARQLWrapper.SPARQLWrapper(getenv('SPARQL_ENDPOINT','http://dbpedia.org/sparql'))
try:
    timeout = max(5, calibrate_query_timeout(sparql))  # 5s for warmup
except IOError:
    from nose import SkipTest
    raise SkipTest(
        "Can't establish connection to SPARQL_ENDPOINT:\n    %s\n"
        "Skipping tests in\n    %s" % (SPARQL_ENDPOINT, __file__))

def test_eval():
    res, matching_node_pairs, gtp_precisions = evaluate(sparql, timeout, gtp_scores_, gp, run=0, gen=0)
    logger.log(
        logging.INFO,
        'Results are:\n'
        'remaining_gain: %d\n'
        'score: %d\n'
        'gain: %d\n'
        'fm: %d\n'
        'avg_res_length: %d\n'
        'sum_gt_matches: %d\n'
        'pattern_length: %d\n'
        'pattern_vars:: %d\n'
        'qtime_exceeded: %d\n'
        'query_time: %d\n'
        % res
    )

def test_mut_fv():
    res = mutate_fix_var(sparql,timeout,gtp_scores_,gp,rand_var=v)
    for gp_ in res:
        logger.info(gp_)

def test_eval_list():
    list = mutate_fix_var(sparql,timeout,gtp_scores_,gp,rand_var=v)
    for gp_ in list:
        res, matching_node_pairs, gtp_precisions = evaluate(sparql, timeout, gtp_scores_, gp_, run=0, gen=0)
        logger.log(
            logging.INFO,
            'For %s\n'
            '%s', gp_,
            'the results are:\n'
            'remaining_gain: %d\n'
            'score: %d\n'
            'gain: %d\n'
            'fm: %d\n'
            'avg_res_length: %d\n'
            'sum_gt_matches: %d\n'
            'pattern_length: %d\n'
            'pattern_vars:: %d\n'
            'qtime_exceeded: %d\n'
            'query_time: %d\n'
            %res
        )

