# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Tested das bauen von graph_pattern per gesampeltem finden von 1-hop wegen
und fix-var-mutation
"""

import logging
import random
from collections import defaultdict
from collections import OrderedDict
from os import getenv

import SPARQLWrapper
from splendid import get_path
from splendid import time_func
import socket
import rdflib
from rdflib import BNode
from rdflib import Literal
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
from utils import sparql_json_result_bindings_to_rdflib

logger = logging.getLogger(__name__)

sparql = SPARQLWrapper.SPARQLWrapper(SPARQL_ENDPOINT)
# sparql = SPARQLWrapper.SPARQLWrapper(
#     getenv('SPARQL_ENDPOINT', 'http://dbpedia.org/sparql'))
try:
    timeout = max(5, calibrate_query_timeout(sparql))  # 5s for warmup
except IOError:
    from nose import SkipTest
    raise SkipTest(
        "Can't establish connection to SPARQL_ENDPOINT:\n    %s\n"
        "Skipping tests in\n    %s" % (sparql.endpoint, __file__))

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

gp_5 = GraphPattern([
    (SOURCE_VAR, a, c),
    (TARGET_VAR, URIRef('http://dbpedia.org/ontology/thumbnail'), d),
    (TARGET_VAR, URIRef('http://dbpedia.org/property/image'), b),
    (c, URIRef('http://dbpedia.org/ontology/wikiPageWikiLink'), SOURCE_VAR),
    (c, URIRef('http://purl.org/linguistics/gold/hypernym'), TARGET_VAR)
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
ground_truth_pairs_2 = random.sample(ground_truth_pairs_2, 100)

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


def test_count(gtps, max_out):
    # values = {(SOURCE_VAR, TARGET_VAR): gtps} hier besser nur die sources
    source_list = [(stp[0], ) for stp in gtps]
    values = {(SOURCE_VAR, ): source_list}
    gp1 = GraphPattern([(SOURCE_VAR, a, b)])
    gp2 = GraphPattern([(b, c, TARGET_VAR)])
    # SPARQL-Query die über eine Var aus gp1 random samplet
    q = gp1.to_sparql_filter_by_count_out_query(
        values=values, count_node=b, max_out=max_out, limit=200)
    logger.info(q)
    t, q_res1 = run_query(q)
    logger.info(q_res1)
    # Kreiere b_list in der die Ergebnisse für b "gespeichert" sind
    # TODO: als Methode, die Listenform (Tupellistenform) der gefundenen
    # Bindings zu gewünschten Variablen zurückgibt.
    res_rows_path = ['results', 'bindings']
    bind1 = sparql_json_result_bindings_to_rdflib(
        get_path(q_res1, res_rows_path, default=[])
    )
    b_list = []
    for row in bind1:
        x = get_path(row, [b])
        y = (x, )
        b_list.append(y)
    logger.info('orig query took %.4f s, result:\n%s\n', t, b_list)
    b_list[:] = [b_l for b_l in b_list if not list_remove_bool(b_l[0])]
    b_list = list(set(b_list))
    # Values für die nächste query: b_list
    values = {(b, ): b_list}
    # Query die über eine var aus gp2 random samplet mit values aus b_list
    q = gp2.to_sparql_select_sample_query(values=values, limit=5000)
    logger.info(q)
    try:
        t, q_res2 = run_query(q)
    except:
        return []
    # Kreiere target_list, in der die "gefundenen" Targets vermerkt sind
    bind2 = sparql_json_result_bindings_to_rdflib(
        get_path(q_res2, res_rows_path, default=[])
    )
    target_list = []
    for row in bind2:
        target_list.append(get_path(row, [TARGET_VAR]))
    logger.info('orig query took %.4f s, result:\n%s\n', t, q_res2)
    # Kreire gtps_2 in der alle gtps, deren targets in target_list enthalten
    # sind, "gespeichert" werden
    gtps_2 = []
    for t in target_list:
        for gtp in gtps:
            if t == gtp[1]:
                gtps_2.append(gtp)
    logger.info(gtps_2)

    # GraphPattern mit gefixten Pfaden aus den gefundenen gtp kreieren:
    # TODO: Das ganze als Methode aus einem graph-pattern, den results und
    # den stp
    gp_list = []
    for row2 in bind2:
        for gtp in gtps:
            if gtp[1] == get_path(row2, [TARGET_VAR]):
                for row1 in bind1:
                    if get_path(row1, [b]) == get_path(row2, [b]):
                        gp_ = GraphPattern([
                            (SOURCE_VAR, get_path(row1, [a]), b),
                            (b, get_path(row2, [c]), TARGET_VAR)
                        ])
                        if gp_ not in gp_list:
                            gp_list.append(gp_)

    # gp3 = GraphPattern([
    #     (SOURCE_VAR, a, b),
    #     (b, c, TARGET_VAR)
    # ])
    gtp_scores = GTPScores(gtps)
    # gtp_scores2 = GTPScores(gtps_2)

    # # Fixe das pattern über die gefundenen gtps
    # mfv2 = []
    # if len(gtps_2) > 1:
    #     mfv2 = mutate_fix_var(sparql, timeout, gtp_scores2, gp3)
    #
    # # lasse die gefundenen Pattern einmal durch die fix_var laufen
    # mfv = []
    # for gp_mfv2 in mfv2:
    #     mfv_res = mutate_fix_var(sparql, timeout, gtp_scores, gp_mfv2)
    #     for gp_res in mfv_res:
    #         mfv.append(gp_res)
    #
    # # evaluiere die so gefundenen Pattern
    # res_eval = eval_gp_list(gtp_scores, mfv)
    # return res_eval

    # evaluiere die gefixten pattern
    res_eval = eval_gp_list(gtp_scores, gp_list)
    return res_eval


def test_sample(gtps):
    values = {(SOURCE_VAR, TARGET_VAR): gtps}
    gp1 = GraphPattern([(SOURCE_VAR, a, b)])
    gp2 = GraphPattern([(b, c, TARGET_VAR)])
    # SPARQL-Query die über eine Var aus gp1 random samplet.
    # TODO: Query so verändern, dass nach count gefiltert wird (siehe log.txt)
    q = gp1.to_sparql_select_sample_query(values=values, limit=100)
    logger.info(q)
    t, q_res1 = run_query(q)
    logger.info(q_res1)
    # Kreiere b_list in der die Ergebnisse für b "gespeichert" sind
    res_rows_path = ['results', 'bindings']
    bind1 = sparql_json_result_bindings_to_rdflib(
        get_path(q_res1, res_rows_path, default=[])
    )
    b_list = []
    for row in bind1:
        x = get_path(row, [b])
        y = (x, )
        b_list.append(y)
    logger.info('orig query took %.4f s, result:\n%s\n', t, b_list)
    b_list[:] = [b_l for b_l in b_list if not list_remove_bool(b_l[0])]
    # Values für die nächste query: b_list
    values = {(b, ): b_list}
    # Query die über eine var aus gp2 random samplet mit values aus b_list
    q = gp2.to_sparql_select_sample_query(values=values, limit=5000)
    logger.info(q)
    t, q_res2 = run_query(q)
    # Kreiere target_list, in der die "gefundenen" Targets vermerkt sind
    bind2 = sparql_json_result_bindings_to_rdflib(
        get_path(q_res2, res_rows_path, default=[])
    )
    target_list = []
    for row in bind2:
        target_list.append(get_path(row, [TARGET_VAR]))
    logger.info('orig query took %.4f s, result:\n%s\n', t, q_res2)
    # Kreire gtps_2 in der alle gtps, deren targets in target_list enthalten
    # sind, "gespeichert" werden
    gtps_2 = []
    for t in target_list:
        for gtp in gtps:
            if t == gtp[1]:
                gtps_2.append(gtp)
    logger.info(gtps_2)

    # GraphPattern mit gefixten Pfaden aus den gefundenen gtp kreieren:
    # TODO: Das ganze als Methode aus einem graph-pattern, den results und
    # den stp
    gp_list = []
    for row2 in bind2:
        for gtp in gtps:
            if gtp[1] == get_path(row2, [TARGET_VAR]):
                for row1 in bind1:
                    if get_path(row1, [b]) == get_path(row2, [b]):
                        gp_ = GraphPattern([
                            (SOURCE_VAR, get_path(row1, [a]), b),
                            (b, get_path(row2, [c]), TARGET_VAR)
                        ])
                        if gp_ not in gp_list:
                            gp_list.append(gp_)

    # gp3 = GraphPattern([
    #     (SOURCE_VAR, a, b),
    #     (b, c, TARGET_VAR)
    # ])
    gtp_scores = GTPScores(gtps)
    # gtp_scores2 = GTPScores(gtps_2)

    # # Fixe das pattern über die gefundenen gtps
    # mfv2 = []
    # if len(gtps_2) > 1:
    #     mfv2 = mutate_fix_var(sparql, timeout, gtp_scores2, gp3)
    #
    # # lasse die gefundenen Pattern einmal durch die fix_var laufen
    # mfv = []
    # for gp_mfv2 in mfv2:
    #     mfv_res = mutate_fix_var(sparql, timeout, gtp_scores, gp_mfv2)
    #     for gp_res in mfv_res:
    #         mfv.append(gp_res)
    #
    # # evaluiere die so gefundenen Pattern
    # res_eval = eval_gp_list(gtp_scores, mfv)
    # return res_eval

    # evaluiere die gefixten pattern
    res_eval = eval_gp_list(gtp_scores, gp_list)
    return res_eval


# Runs a given (as String) query against the Sparql-endpoint
def run_query(q):
    try:
        q_short = ' '.join((line.strip() for line in q.split('\n')))
        sparql.setQuery(q_short)
        cal = time_func(sparql.queryAndConvert)
    except socket.timeout:
        cal = (timeout, {})
    except ValueError:
        # e.g. if the endpoint gives us bad JSON for some unicode chars
        logger.info(
            'Could not parse result for query, assuming empty result...\n'
            'Query:\n%s\nException:', q,
            exc_info=1,  # appends exception to message
        )
        cal = (timeout, {})
    return cal


# Checks if an found RDF-Term can be used as value in a new query
# (without conflicts)
def list_remove_bool(var):
    if isinstance(var, Literal):
        i_n3 = var.n3()
        if len(i_n3) > 60:
            return True
    elif isinstance(var, BNode):
        return True
    # echt hässlich, aber die einzige Möglichkeit, die ich gesehen habe um
    # keine Probleme mit dem Category:Cigarettes-Beispiel zu bekommen
    # (siehe docs)
    # TODO: Möglicherweise dafür sorgen, dass die nicht rausgeschmissen,
    # sondern nur nicht mit prefix gekürzt werden, also einfach mal schauen,
    # dass die curify das tut was sie soll
    elif isinstance(var, URIRef):
        return ':' in var[7:]
    return False


# evaluates a given graph-pattern-list
def eval_gp_list(gtp_scores, gp_list):
    for gp_l in gp_list:
        res_ev = evaluate(
            sparql, timeout, gtp_scores, gp_l, run=0, gen=0)
        update_individuals([gp_l], [res_ev])
        # print_graph_pattern(gp_, print_matching_node_pairs=0)
    return gp_list


if __name__ == '__main__':
    # # test_sample:
    # res = []
    # for i in range(10):
    #     res_ts = test_sample(ground_truth_pairs_2)
    #     for gp_ts in res_ts:
    #         res.append(gp_ts)
    #
    # res = sorted(res, key=lambda gp_: -gp_.fitness.values.score)
    # for res_ in res:
    #     print_graph_pattern(res_)

    # test_count
    res = []
    for i in range(1):
        ground_truth_pairs_5 = get_semantic_associations()
        ground_truth_pairs_5 = random.sample(ground_truth_pairs_5, 200)
        max_out_steps = [10, 15, 20, 25, 30, 40, 50, 75, 100]
        for j in max_out_steps:
            res_ts = test_count(ground_truth_pairs_5, j)
            for gp_ts in res_ts:
                res.append((gp_ts, j))

    res = sorted(res, key=lambda gp_: -gp_[0].fitness.values.score)
    res = res[0:100]
    for res_ in res:
        print('max_out:'+str(res_[1]))
        print_graph_pattern(res_[0])
