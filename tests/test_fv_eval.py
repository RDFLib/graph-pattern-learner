# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""test_mutate_fix_var und test_evaluate einmal davor und 
einmal über die results aus mutate_fix_var
"""

import logging
from collections import OrderedDict
from os import getenv

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
from serialization import print_graph_pattern

logger = logging.getLogger(__name__)

dbp = rdflib.Namespace('http://dbpedia.org/resource/')
owl = rdflib.Namespace('http://www.w3.org/2002/07/owl#')

a = Variable('a')
b = Variable('b')
c = Variable('c')
d = Variable('d')
e = Variable('e')
f = Variable('f')
v = Variable('v')
w = Variable('w')

sameAs = owl['sameAs']

gp_1 = GraphPattern([
    (SOURCE_VAR, v, TARGET_VAR)
])

gp_2 = GraphPattern([
    (SOURCE_VAR, v, TARGET_VAR),
    (TARGET_VAR, w, SOURCE_VAR)
])

gp_3 = GraphPattern([
    (SOURCE_VAR, a, b),
    (b, c, d),
    (d, e, TARGET_VAR)
])

gp_4 = GraphPattern([
    (SOURCE_VAR, a, b),
    (b, c, d),
    (TARGET_VAR, e, d)
])

ground_truth_pairs_1 = [
    (dbp['Berlin'], dbp['Germany']),
    (dbp['Hamburg'], dbp['Germany']),
    (dbp['Kaiserslautern'], dbp['Germany']),
    (dbp['Wien'], dbp['Austria']),
    (dbp['Insbruck'], dbp['Austria']),
    (dbp['Salzburg'], dbp['Austria']),
    (dbp['Paris'], dbp['France']),
    (dbp['Lyon'], dbp['France']),
    (dbp['Amsterdam'], dbp['Netherlands']),
    (dbp['Brussels'], dbp['Belgium']),
    (dbp['Washington'], dbp['United_States']),
    (dbp['Madrid'], dbp['Spain']),
    (dbp['Prague'], dbp['Czech_Republic']),
    (dbp['Bern'], dbp['Switzerland']),
]

ground_truth_pairs_2 = get_semantic_associations()
ground_truth_pairs_2, _ = split_training_test_set(ground_truth_pairs_2)
ground_truth_pairs_2 = ground_truth_pairs_2[1:10]

ground_truth_pairs_3 = [
    (dbp['Barrister'], dbp['Law']),
    (dbp['Christ'], dbp['Jesus']),
    (dbp['Pottage'], dbp['Soup'])
    ]

ground_truth_pairs_4 = [
    (dbp['Motorrad_(disambiguation)'], dbp['Bmw_motorcycle']),
    (dbp['Horse'], dbp['Saddle'])
]

gtp_scores_1 = GTPScores(ground_truth_pairs_1)
gtp_scores_2 = GTPScores(ground_truth_pairs_2)
gtp_scores_3 = GTPScores(ground_truth_pairs_3)
gtp_scores_4 = GTPScores(ground_truth_pairs_4)

sparql = SPARQLWrapper.SPARQLWrapper(
    getenv('SPARQL_ENDPOINT', 'http://dbpedia.org/sparql'))
try:
    timeout = max(5, calibrate_query_timeout(sparql))  # 5s for warmup
except IOError:
    from nose import SkipTest
    raise SkipTest(
        "Can't establish connection to SPARQL_ENDPOINT:\n    %s\n"
        "Skipping tests in\n    %s" % (SPARQL_ENDPOINT, __file__))


def test_eval(gtp_scores, gp):
    res, matching_node_pairs, gtp_precisions = evaluate(
        sparql, timeout, gtp_scores, gp, run=0, gen=0)
    update_individuals([gp], [(res, matching_node_pairs, gtp_precisions)])
    logger.info(gp.fitness)


def test_mut_fv(gtp_scores, gp, r=None):
    res = mutate_fix_var(sparql, timeout, gtp_scores, gp, rand_var=r)
    for gp_ in res:
        logger.info(gp_)


def test_eval_list(gtp_scores, gp, r=None):
    mfv_res = mutate_fix_var(sparql, timeout, gtp_scores, gp, rand_var=r)
    for gp_ in mfv_res:
        res, matching_node_pairs, gtp_precisions = evaluate(
            sparql, timeout, gtp_scores, gp_, run=0, gen=0)
        update_individuals([gp_], [(res, matching_node_pairs, gtp_precisions)])
        print_graph_pattern(gp_, print_matching_node_pairs=0)
    return mfv_res


def test_eval_list_double(gtp_scores, gp, r_1=None, r_2=None):
    # testing double execution of mutate_fix_var() on gp
    res = test_eval_list(gtp_scores, gp, r_1)
    gtp_scores.update_with_gps(res)
    res_list = list(res)
    for gp in res:
        res_ = test_eval_list(gtp_scores, gp, r_2)
        for gp_ in res_:
            res_list.append(gp_)
    gtp_scores.update_with_gps(res_list)
    for gp in res_list:
        print_graph_pattern(gp, print_matching_node_pairs=0)


if __name__ == '__main__':
    #test_eval_list_double(gtp_scores_1, gp_2)

    test_eval_list_double(gtp_scores_4, gp_4, a, e)
