# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Testet die verschiedenen Versionen der mutatete_deep_narrow
"""

import logging
import numpy as np
import pickle
import random
from collections import defaultdict
from collections import OrderedDict
from os import getenv

import SPARQLWrapper
from itertools import chain
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
from gp_learner import mutate_deep_narrow_path
from gp_learner import mutate_fix_var
from gp_learner import update_individuals
from gp_query import calibrate_query_timeout
from gp_query import query_time_hard_exceeded
from gp_query import query_time_soft_exceeded
from graph_pattern import gen_random_var
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

dbr = rdflib.Namespace('http://dbpedia.org/resource/')
owl = rdflib.Namespace('http://www.w3.org/2002/07/owl#')
dbo = rdflib.Namespace('http://dbpedia.org/ontology/')
gold = rdflib.Namespace('http://purl.org/linguistics/gold')
dbt = rdflib.Namespace('http://dbpedia.org/resource/Template:')
dbp = rdflib.Namespace('http://dbpedia.org/property/')

v = [gen_random_var() for i in range(100)]

sameAs = owl['sameAs']
pwl = dbo['wikiPageWikiLink']
hypernym = gold['hypernym']
wpUseTemp = dbp['wikiPageUsesTemplate']

gp_found = {}
gp_found['1'] = GraphPattern([
    (SOURCE_VAR, pwl, TARGET_VAR),
    (SOURCE_VAR, v[0], v[1]),
    (v[1], hypernym, TARGET_VAR)
])
gp_found['2'] = GraphPattern([
    (SOURCE_VAR, pwl, TARGET_VAR),
    (TARGET_VAR, v[0], SOURCE_VAR),
    (TARGET_VAR, v[1], URIRef('http://dbpedia.org/dbtax/Page'))
])
gp_found['3'] = GraphPattern([
    (SOURCE_VAR, pwl, TARGET_VAR),
    (TARGET_VAR, v[0], SOURCE_VAR),
    (TARGET_VAR, v[1], dbt['Sister_project_links'])
])
gp_found['4'] = GraphPattern([
    (SOURCE_VAR, pwl, TARGET_VAR),
    (TARGET_VAR, wpUseTemp, dbt['Pp-semi-indef'])
])
gp_found['5'] = GraphPattern([
    (SOURCE_VAR, pwl, TARGET_VAR),
    (TARGET_VAR, v[0], dbt['Pp-semi-indef'])
])
gp_found['6'] = GraphPattern([
    (SOURCE_VAR, pwl, TARGET_VAR),
    (TARGET_VAR, v[0], SOURCE_VAR),
    (TARGET_VAR, v[1], dbt['Cite_book'])
])
gp_found['7'] = GraphPattern([
    (SOURCE_VAR, pwl, TARGET_VAR),
    (TARGET_VAR, v[0], SOURCE_VAR),
    (TARGET_VAR, v[1], dbt['Redirect'])
])
gp_found['8'] = GraphPattern([
    (SOURCE_VAR, hypernym, TARGET_VAR)
])
gp_found['50'] = GraphPattern([
    (SOURCE_VAR, pwl, TARGET_VAR),
    (TARGET_VAR, v[0], SOURCE_VAR),
    (TARGET_VAR, v[1], dbt['Use_dmy_dates'])
])
gp_found['51'] = GraphPattern([
    (SOURCE_VAR, pwl, TARGET_VAR),
    (TARGET_VAR, v[0], SOURCE_VAR),
    (TARGET_VAR, v[1], dbt['Refend'])
])
gp_found['52'] = GraphPattern([
    (SOURCE_VAR, pwl, TARGET_VAR),
    (TARGET_VAR, URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'), 
     URIRef('http://dbpedia.org/dbtax/Page'))
])
gp_found['54'] = GraphPattern([
    (SOURCE_VAR, hypernym, TARGET_VAR),
    (v[0], sameAs, SOURCE_VAR)
])
gp_found['55'] = GraphPattern([
    (SOURCE_VAR, hypernym, TARGET_VAR),
    (TARGET_VAR, pwl, SOURCE_VAR)
])
gp_found['67'] = GraphPattern([
    (SOURCE_VAR, pwl, TARGET_VAR),
    (TARGET_VAR, v[0], SOURCE_VAR),
    (TARGET_VAR, v[1], dbt['Portal'])
])
gp_found['68'] = GraphPattern([
    (SOURCE_VAR, pwl, TARGET_VAR),
    (TARGET_VAR, v[0], SOURCE_VAR),
    (TARGET_VAR, v[1], dbt['Convert'])
])
gp_found['69'] = GraphPattern([
    (SOURCE_VAR, hypernym, TARGET_VAR),
    (v[0], hypernym, SOURCE_VAR)
])
gp_found['72'] = GraphPattern([
    (SOURCE_VAR, URIRef('http://purl.org/dc/terms/subject'), v[1]),
    (TARGET_VAR, pwl, SOURCE_VAR),
    (v[0], sameAs, v[1]),
    (v[1], URIRef('http://www.w3.org/2004/02/skos/core#subject'), TARGET_VAR)
])
gp_found['94'] = GraphPattern([
    (SOURCE_VAR, URIRef('http://purl.org/dc/terms/subject'), v[1]),
    (TARGET_VAR, v[0], SOURCE_VAR),
    (v[1], URIRef('http://www.w3.org/2004/02/skos/core#subject'), TARGET_VAR)
])
gp_found['131'] = GraphPattern([
    (SOURCE_VAR, v[0], v[2]),
    (TARGET_VAR, pwl, v[1]),
    (v[2], URIRef('http://www.w3.org/2004/02/skos/core#subject'), TARGET_VAR),
])
gp_found['140'] = GraphPattern([
    (TARGET_VAR, pwl, SOURCE_VAR),
    (TARGET_VAR, wpUseTemp, dbt['Other_uses']),
    (TARGET_VAR, wpUseTemp, dbt['Pp-move-indef']),
    (v[0], URIRef('http://www.w3.org/2000/01/rdf-schema#seeAlso'), TARGET_VAR),
])
# Bis hier jedes mit neuem Fingerprint, jetzt noch 3 vom Rest
gp_found['231'] = GraphPattern([
    (SOURCE_VAR, dbo['class'], TARGET_VAR),
    (TARGET_VAR, dbp['subdivisionRanks'], v[0])
])
gp_found['323'] = GraphPattern([
    (SOURCE_VAR, pwl, TARGET_VAR),
    (v[0], dbp['species'], TARGET_VAR),
    (v[1], dbo['wikiPageDisambiguates'], TARGET_VAR)
])
gp_found['516'] = GraphPattern([
    (SOURCE_VAR, pwl, v[1]),
    (TARGET_VAR, dbp['image'], v[0]),
    (v[1], hypernym, TARGET_VAR),
    (v[2], dbo['wikiPageRedirects'], SOURCE_VAR)
])

# Verschiedene Limits festlegen:
# Limit: search object-list => subject-values in next query
limit_next = 500
# limt: search an object list from two diferrent subjects and get hits through
# comparing them
limit_endpoint_two_sided = 1000
# limit: search object-list => compare with sources/targets from gtp
limit_choose_endpoint = 5000
# limit: search subject-list from two diferrent objects and get hits through
# comparing them
limit_startpoint_two_sided = 200
# limit: search subject-list => subject-values in next query
limit_subject_next = 350
# limit: search subject list => compare with sources/targets from gtp
limit_choose_subject_endpoint = 3000
# limits: hit-list => on side subject, one side object:
limit_subj_to_obj = 350
limit_obj_to_subj = 1500


# einen ein-hop-weg von source zu target zum pattern hinzufügen
# TODO Varianten (von gefundenen b aus Variante der zweiten query
# 1.(default) mit (b, c, d) Liste von d suchen und mit Target-Liste vergleichen
# 2. mit (b, c, target). VALUES(target) suchen =>
# Ergebnisse direkt an existente Targets gebunden
# 3. mit (b, c, target).urspurngs_gp
def mutate_deep_narrow_one_hop_s_t_without_direction(
        gp_, gtps, max_out=None, max_in=None, in_out=None
):
    vars_ = gp_.vars_in_graph
    if not (SOURCE_VAR in vars_ and TARGET_VAR in vars_):
        logger.info('SOURCE or TARGET are not in gp: %s' % gp_)
        return []
    # Erstelle pattern für den ersten Schritt
    a = Variable('a')
    b = Variable('b')
    c = Variable('c')
    values_s_t = {(SOURCE_VAR, TARGET_VAR): gtps}
    gp1 = GraphPattern([(SOURCE_VAR, a, b)])
    q = gp1.to_sparql_filter_by_count_in_out_query(
        values=values_s_t, count_node=b, in_out=in_out, max_out=max_out,
        max_in=max_in, gp=gp_, limit=200)
    logger.info(q)
    t, q_res1 = run_query(q)
    if not q_res1['results']['bindings']:
        return []
    # logger.info('orig query took %.4f s, result:\n%s\n', t, q_res1)
    # Erstelle values aus den Ergebnissen für b
    values = get_values([b], q_res1)
    gp2 = GraphPattern([(b, c, TARGET_VAR)])
    # Query die über eine var aus gp2 random samplet mit values aus b_list
    q = gp2.to_sparql_select_sample_query(values=values, limit=5000)
    logger.info(q)
    try:
        t, q_res2 = run_query(q)
    except:
        logger.info('Die Query (s.o.) hat nicht geklappt')
        return []
    # Kreiere target_list, in der die "gefundenen" Targets vermerkt sind
    target_list = get_values_list(TARGET_VAR, q_res2)
    # logger.info('orig query took %.4f s, result:\n%s\n', t, q_res2)
    # Kreiere gtps_hit in der alle gtps, deren targets in target_list enthalten
    # sind, "gespeichert" werden
    stp_hit = get_stp_hit(target_list, gtps, 1)
    gp_list = get_fixed_path_gp_one_hop(
        q_res1, q_res2, gp_, stp_hit, [], a, b, c
    )
    return gp_list


# einen ein-hop-weg von source zu target zum pattern hinzufügen 
# (gp in query 2 eingefügt)
def mutate_deep_narrow_one_hop_s_t_2(gp_, gtps, max_in_out=None, in_out=None):
    vars_ = gp_.vars_in_graph
    if not (SOURCE_VAR in vars_ and TARGET_VAR in vars_):
        logger.info('SOURCE or TARGET are not in gp: %s' % gp_)
        return []
    # Erstelle pattern für den ersten Schritt
    a = Variable('a')
    b = Variable('b')
    c = Variable('c')
    gp1 = GraphPattern([(SOURCE_VAR, a, b)])
    values_s_t = {(SOURCE_VAR, TARGET_VAR): gtps}
    q = gp1.to_sparql_filter_by_count_in_out_query(
        values=values_s_t, count_node=b, in_out=in_out, 
        max_out=max_in_out, gp=gp_, limit=200)
    logger.info(q)
    t, q_res1 = run_query(q)
    if not q_res1['results']['bindings']:
        return []
    # logger.info('orig query took %.4f s, result:\n%s\n', t, q_res1)
    gp2 = GraphPattern([(b, c, TARGET_VAR)])
    # Erstelle values aus den Ergebnissen für b
    values = get_values([b], q_res1)
    # Query die über eine var aus gp2 random samplet mit values aus b_list
    q = gp2.to_sparql_select_sample_query(
        values=values, values_s_t=values_s_t, limit=5000
    )
    logger.info(q)
    try:
        t, q_res2 = run_query(q)
    except:
        logger.info('Die Query (s.o.) hat nicht geklappt')
        return []
    # Kreiere target_list, in der die "gefundenen" Targets vermerkt sind
    target_list = get_values_list(TARGET_VAR, q_res2)
    # logger.info('orig query took %.4f s, result:\n%s\n', t, q_res2)
    # Kreiere gtps_hit in der alle gtps, deren targets in target_list enthalten
    # sind, "gespeichert" werden
    stp_hit = get_stp_hit(target_list, gtps, 1)
    gp_list = get_fixed_path_gp_one_hop(q_res1, q_res2, gp_, stp_hit, a, b, c)
    return gp_list


# eine one-hop verbindung zwischen source und target finden (Richtungen random)
def mutate_deep_narrow_one_random_hop_s_t():
    ich_darf_nich_leer_sein = []
    return ich_darf_nich_leer_sein


# einen direkten weg um einen hop erweitern (Weg löschen und stattdessen
# ein-hop weg einfügen)


# zu einem direkten weg noch einen ein-hop weg hinzufügen (weg behalten,
# ein-hop weg dazu)


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


# returns a list of value-tupels for the given variables, out of an
# query-result
def get_values(varlist, q_res):
    res_rows_path = ['results', 'bindings']
    bind = sparql_json_result_bindings_to_rdflib(
        get_path(q_res, res_rows_path, default=[])
    )
    vallist = []
    for row in bind:
        tup = ()
        for var in varlist:
            tup = tup + (get_path(row, [var]), )
        vallist.append(tup)
    # ausfiltern von vallist (leider notwendig vor allem wegen dbr:Template
    vallist[:] = [valtup for valtup in vallist if not list_remove_bool(valtup)]
    # dopppelte noch herausfiltern
    vallist = list(set(vallist))
    vartup = ()
    for var in varlist:
        vartup = vartup + (var, )
    values = {vartup: vallist}
    return values


# returns a list of found values for a given variable and query-result
def get_values_list(var, q_res):
    res_rows_path = ['results', 'bindings']
    bind = sparql_json_result_bindings_to_rdflib(
        get_path(q_res, res_rows_path, default=[])
    )
    vallist = [get_path(row, [var]) for row in bind]
    return vallist


# gibt ein sample nach der Gewichtung der counts zurück,
# Gewichtung ist hier innerhalb angesetzt
def get_weighted_sample(var, count, q_res):
    res_rows_path = ['results', 'bindings']
    bind = sparql_json_result_bindings_to_rdflib(
        get_path(q_res, res_rows_path, default=[])
    )
    val = []
    weight = []
    for row in bind:
        val.append(get_path(row, [var]))
        # Davon ausgehend, dass x besonders gut ist
        if float(get_path(row, [count])) == 1.0:
            weight.append(10000)
        else:
            weight.append(1/(abs(1-float(get_path(row, [count])))))
        # Davon ausgehend, dass x besonders schlecht ist
        # weight.append(abs(7-float(get_path(row, [count]))))
        # weight.append(get_path(row, [count]))
    s = sum(weight)
    for i in range(len(weight)):
        weight[i] = weight[i] / s
    cum_weights = [0] + list(np.cumsum(weight))
    res = []
    while len(res) < min(10, len(list(set(val)))):
        x = np.random.random()
        i = 0
        while x > cum_weights[i]:
            i = i + 1
        index = i - 1
        if val[index] not in res:
            res.append((val[index],))
    sample = {(var,): res}
    return sample


# gibt zu einer gegebenen Liste von Variablen die stp aus gtps zurück,
# bei denen Target(st=1)/Source(st=0) in der Variablen Liste ist.
def get_stp_hit(varlist, gtps, st):
    stp = []
    for t in varlist:
        for gtp in gtps:
            if t == gtp[st]: 
                stp.append(gtp)
    return stp


# Checks if an found RDF-Term can be used as value in a new query
# (without conflicts)
def list_remove_bool(tup):
    for var in tup:
        if isinstance(var, Literal):
            i_n3 = var.n3()
            if len(i_n3) > 60:
                return True
        elif isinstance(var, BNode):
            return True
        elif isinstance(var, URIRef):
            return '%' in var
        # TODO: nochmal schauen das % rauswerfen war kuzfristig,
        # weil sparql mir bei einer query nen Fehler geschmissen hat
    return False


# evaluates a given graph-pattern-list
def eval_gp_list(gtp_scores, gp_list):
    for gp_l in gp_list:
        eval_gp(gtp_scores, gp_l)
    return gp_list


# evaluate a given graph-pattern
def eval_gp(gtp_scores, gp):
    res = evaluate(
        sparql, timeout, gtp_scores, gp, run=0, gen=0)
    update_individuals([gp], [res])


# helper to get target-hits and the corresponding stp
def target_hit(stps, t_lis):
    res = []
    for stp in stps:
        for t in t_lis:
            if t == stp[1]:
                res.append(
                    (t, stp)
                )
    return res


# add one hop with the given direction.
def mutate_deep_narrow_one_hop(
        gp_, max_out=None, max_in=None, in_out=None, richtung=None
):
    vars_ = gp_.vars_in_graph
    if not (SOURCE_VAR in vars_ and TARGET_VAR in vars_):
        logger.info('SOURCE or TARGET are not in gp: %s' % gp_)
        return []
    if not gp_.matching_node_pairs:
        logger.info(
            'No matching node pairs, cant get better through adding constraints'
        )
        return []
    # Erstelle pattern für den ersten Schritt
    a = Variable('a')
    b = Variable('b')
    c = Variable('c')
    if richtung not in [1, 2, 3, 4]:
        richtung = random.choice([1, 2, 3, 4])
        logger.info('Richtung %s wurde gewaehlt' % richtung)
    if richtung == 1:
        values_s_t = {(SOURCE_VAR, TARGET_VAR): gp_.matching_node_pairs}
        gp1 = GraphPattern([(SOURCE_VAR, a, b)])
        q = gp1.to_sparql_filter_by_count_in_out_query(
            values=values_s_t, count_node=b, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=200)
        logger.info(q)
        t, q_res1 = run_query(q)
        if not q_res1:
            return []
        # logger.info('orig query took %.4f s, result:\n%s\n', t, q_res1)
        # Erstelle values aus den Ergebnissen für b
        values = get_values([b], q_res1)
        gp2 = GraphPattern([(b, c, TARGET_VAR)])
        # Query die über eine var aus gp2 random samplet mit values aus b_list
        q = gp2.to_sparql_select_sample_query(values=values, limit=5000)
        logger.info(q)
        try:
            t, q_res2 = run_query(q)
        except:
            logger.info('Die Query (s.o.) hat nicht geklappt')
            return []
        # logger.info('orig query took %.4f s, result:\n%s\n', t, q_res2)
        gp_list = get_fixed_path_gp_one_hop(
            q_res1, q_res2, gp_, richtung, gp_.matching_node_pairs, a, b, c
        )
    elif richtung == 2:
        values_s = {
            (SOURCE_VAR, ): [(tup[0], ) for tup in gp_.matching_node_pairs]
        }
        values_t = {
            (TARGET_VAR, ): [(tup[1], ) for tup in gp_.matching_node_pairs]
        }
        gp1 = GraphPattern([(SOURCE_VAR, a, b)])
        gp2 = GraphPattern([(TARGET_VAR, c, b)])
        q = gp1.to_sparql_filter_by_count_in_out_query(
            values=values_s, count_node=b, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=1000)
        logger.info(q)
        t, q_res1 = run_query(q)
        if not q_res1['results']['bindings']:
            return []
        q = gp2.to_sparql_filter_by_count_in_out_query(
            values=values_t, count_node=b, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=1000)
        logger.info(q)
        t, q_res2 = run_query(q)
        if not q_res2['results']['bindings']:
            return []
        gp_list = get_fixed_path_gp_one_hop(
            q_res1, q_res2, gp_, richtung, gp_.matching_node_pairs, a, b, c
        )
    elif richtung == 3:
        values_s_t = {(SOURCE_VAR, TARGET_VAR): gp_.matching_node_pairs}
        gp2 = GraphPattern([(TARGET_VAR, c, b)])
        q = gp2.to_sparql_filter_by_count_in_out_query(
            values=values_s_t, count_node=b, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=200)
        logger.info(q)
        t, q_res2 = run_query(q)
        if not q_res2['results']['bindings']:
            return []
        # logger.info('orig query took %.4f s, result:\n%s\n', t, q_res1)
        gp1 = GraphPattern([(b, a, SOURCE_VAR)])
        # Erstelle values aus den Ergebnissen für b
        values = get_values([b], q_res2)
        # Query die über eine var aus gp2 random samplet mit values aus b_list
        q = gp1.to_sparql_select_sample_query(values=values, limit=5000)
        logger.info(q)
        try:
            t, q_res1 = run_query(q)
        except:
            logger.info('Die Query (s.o.) hat nicht geklappt')
            return []
        gp_list = get_fixed_path_gp_one_hop(
            q_res1, q_res2, gp_, richtung, gp_.matching_node_pairs, a, b, c
        )
    else:
        values_s = {
            (SOURCE_VAR, ): [(tup[0], ) for tup in gp_.matching_node_pairs]
        }
        values_t = {
            (TARGET_VAR, ): [(tup[1], ) for tup in gp_.matching_node_pairs]
        }
        gp1 = GraphPattern([(b, a, SOURCE_VAR)])
        gp2 = GraphPattern([(b, c, TARGET_VAR)])
        q = gp1.to_sparql_filter_by_count_in_out_query(
            values=values_s, count_node=b, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=200)
        logger.info(q)
        t, q_res1 = run_query(q)
        if not q_res1['results']['bindings']:
            return []
        q = gp2.to_sparql_filter_by_count_in_out_query(
            values=values_t, count_node=b, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=200)
        logger.info(q)
        t, q_res2 = run_query(q)
        if not q_res2['results']['bindings']:
            return []
        gp_list = get_fixed_path_gp_one_hop(
            q_res1, q_res2, gp_, richtung, gp_.matching_node_pairs, a, b, c
        )
    return gp_list


# fixed den ein-hop-pfad zwischen Source und Target, fügt ihn dem Pattern hinzu
# und gibt die Liste der resultierenden Pattern zurück
# TODO nicht so sehr auf source a b. b c Target fokussieren.
def get_fixed_path_gp_one_hop(q_res1, q_res2, gp_main, richtung, stp, a, b, c):
    gp_list = []
    res_rows_path = ['results', 'bindings']
    bind1 = sparql_json_result_bindings_to_rdflib(
        get_path(q_res1, res_rows_path, default=[])
    )
    bind2 = sparql_json_result_bindings_to_rdflib(
        get_path(q_res2, res_rows_path, default=[])
    )
    for row2 in bind2:
        for gtp in stp:
            if gtp[1] == get_path(row2, [TARGET_VAR]):
                for row1 in bind1:
                    if (get_path(row1, [b]) == get_path(row2, [b])) and \
                            (get_path(row1, [SOURCE_VAR]) == gtp[0]):
                        if richtung == 1:
                            gp_ = GraphPattern([
                                (SOURCE_VAR, get_path(row1, [a]), b),
                                (b, get_path(row2, [c]), TARGET_VAR)
                            ])
                        elif richtung == 2:
                            gp_ = GraphPattern([
                                (SOURCE_VAR, get_path(row1, [a]), b),
                                (TARGET_VAR, get_path(row2, [c]), b)
                            ])
                        elif richtung == 3:
                            gp_ = GraphPattern([
                                (b, get_path(row1, [a]), SOURCE_VAR),
                                (TARGET_VAR, get_path(row2, [c]), b)
                            ])
                        else:
                            gp_ = GraphPattern([
                                (b, get_path(row1, [a]), SOURCE_VAR),
                                (b, get_path(row2, [c]), TARGET_VAR)
                            ])

                        gp_ = GraphPattern(chain(gp_, gp_main))
                        if gp_ not in gp_list:
                            gp_list.append(gp_)
                        logger.info(gtp)
    return gp_list


# fixed den ein-hop-pfad zwischen Source und Target, fügt ihn dem Pattern hinzu
# und gibt die Liste der resultierenden Pattern zurück
# TODO nicht so sehr auf source a b. b c Target fokussieren.
def get_fixed_path_gp_two_hops(
        q_res1, q_res2, q_res3, gp_main, richtung, stp, a, b, c, d, e
):
    # TODO: überlegen nicht nur verschieden Pattern für verschiedene Richtungen
    # zu machen, sondern auch in den Unterschiedlichen Ergebnissn anfangen
    # (Idee wäre z.B. die a bis e durch nummerierte random vars zu ersetzen und
    # sich dann zu überlegen wie man das übergibt, ob mans iwie immer entlang
    # des patterns schafft oder eher nicht.
    gp_list = []
    res_rows_path = ['results', 'bindings']
    bind1 = sparql_json_result_bindings_to_rdflib(
        get_path(q_res1, res_rows_path, default=[])
    )
    bind2 = sparql_json_result_bindings_to_rdflib(
        get_path(q_res2, res_rows_path, default=[])
    )
    bind3 = sparql_json_result_bindings_to_rdflib(
        get_path(q_res3, res_rows_path, default=[])
    )
    for gtp in stp:
        for row3 in bind3:
            if gtp[1] == get_path(row3, [TARGET_VAR]):
                for row2 in bind2:
                    if get_path(row2, [d]) == get_path(row3, [d]):
                        for row1 in bind1:
                            if get_path(row1, [b]) == \
                                    get_path(row2, [b]) and \
                                    get_path(row1, [SOURCE_VAR]) == \
                                    gtp[0]:
                                if richtung == 1:
                                    gp_ = GraphPattern([
                                        (SOURCE_VAR, get_path(row1, [a]), b),
                                        (b, get_path(row2, [c]), d),
                                        (d, get_path(row3, [e]), TARGET_VAR)
                                    ])
                                elif richtung == 2:
                                    gp_ = GraphPattern([
                                        (SOURCE_VAR, get_path(row1, [a]), b),
                                        (b, get_path(row2, [c]), d),
                                        (TARGET_VAR, get_path(row3, [e]), d)
                                    ])
                                elif richtung == 3:
                                    gp_ = GraphPattern([
                                        (SOURCE_VAR, get_path(row1, [a]), b),
                                        (d, get_path(row2, [c]), b),
                                        (d, get_path(row3, [e]), TARGET_VAR)
                                    ])
                                elif richtung == 4:
                                    gp_ = GraphPattern([
                                        (SOURCE_VAR, get_path(row1, [a]), b),
                                        (d, get_path(row2, [c]), b),
                                        (TARGET_VAR, get_path(row3, [e]), d)
                                    ])
                                elif richtung == 5:
                                    gp_ = GraphPattern([
                                        (b, get_path(row1, [a]), SOURCE_VAR),
                                        (b, get_path(row2, [c]), d),
                                        (d, get_path(row3, [e]), TARGET_VAR)
                                    ])
                                elif richtung == 6:
                                    gp_ = GraphPattern([
                                        (b, get_path(row1, [a]), SOURCE_VAR),
                                        (b, get_path(row2, [c]), d),
                                        (TARGET_VAR, get_path(row3, [e]), d)
                                    ])
                                elif richtung == 7:
                                    gp_ = GraphPattern([
                                        (b, get_path(row1, [a]), SOURCE_VAR),
                                        (d, get_path(row2, [c]), b),
                                        (d, get_path(row3, [e]), TARGET_VAR)
                                    ])
                                else:
                                    gp_ = GraphPattern([
                                        (b, get_path(row1, [a]), SOURCE_VAR),
                                        (d, get_path(row2, [c]), b),
                                        (TARGET_VAR, get_path(row3, [e]), d)
                                    ])
                                gp_ = GraphPattern(chain(gp_, gp_main))
                                if gp_ not in gp_list:
                                    gp_list.append(gp_)
                                logger.debug(gtp)
    return gp_list


# add two hops.
def mutate_deep_narrow_two_hops(
        gp_, max_out=None, max_in=None, in_out=None, richtung=None
):
    vars_ = gp_.vars_in_graph
    if not (SOURCE_VAR in vars_ and TARGET_VAR in vars_):
        logger.debug('SOURCE or TARGET are not in gp: %s' % gp_)
        return []
    if not gp_.matching_node_pairs:
        logger.debug(
            'No matching node pairs, cant get better through adding constraints'
        )
        return []
    a = Variable('a')
    b = Variable('b')
    c = Variable('c')
    d = Variable('d')
    e = Variable('e')
    gp_list = []
    if richtung not in range(1, 9):
        richtung = random.choice(range(1, 9))
        logger.debug('Richtung %s wurde gewaehlt' % richtung)
    if richtung == 1:
        gp1 = GraphPattern([(SOURCE_VAR, a, b)])
        gp2 = GraphPattern([(b, c, d)])
        gp3 = GraphPattern([(d, e, TARGET_VAR)])
        
        values_s = {
            (SOURCE_VAR, ): [(tup[0], ) for tup in gp_.matching_node_pairs]
        }
        q = gp1.to_sparql_filter_by_count_in_out_query(
            values=values_s, count_node=b, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_next)
        logger.debug(q)
        try:
            t, q_res1 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res1:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res1['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []
        
        values_b = get_values([b], q_res1)
        q = gp2.to_sparql_filter_by_count_in_out_query(
            values=values_b, count_node=d, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_next)
        logger.debug(q)
        try:
            t, q_res2 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res2:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res2['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []
        
        values_d = get_values([d], q_res2)
        q = gp3.to_sparql_select_sample_query(
            values=values_d, limit=limit_choose_endpoint
        )
        logger.debug(q)
        try:
            t, q_res3 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res3:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res3['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []
        
        gp_list = get_fixed_path_gp_two_hops(
            q_res1,
            q_res2,
            q_res3,
            gp_,
            richtung,
            gp_.matching_node_pairs,
            a,
            b,
            c,
            d,
            e
        )
    if richtung == 2:
        gp1 = GraphPattern([(SOURCE_VAR, a, b)])
        gp2 = GraphPattern([(b, c, d)])
        gp3 = GraphPattern([(TARGET_VAR, e, d)])
        
        values_s = {
            (SOURCE_VAR, ): [(tup[0], ) for tup in gp_.matching_node_pairs]
        }
        q = gp1.to_sparql_filter_by_count_in_out_query(
            values=values_s, count_node=b, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_next)
        logger.debug(q)
        try:
            t, q_res1 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res1:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res1['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []
            
        values_b = get_values([b], q_res1)
        q = gp2.to_sparql_filter_by_count_in_out_query(
            values=values_b, count_node=d, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_endpoint_two_sided)
        logger.debug(q)
        try:
            t, q_res2 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res2:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res2['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []
        
        values_t = {
            (TARGET_VAR, ): [(tup[1], ) for tup in gp_.matching_node_pairs]
        }
        q = gp3.to_sparql_filter_by_count_in_out_query(
            values=values_t, count_node=d, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_endpoint_two_sided)
        logger.debug(q)
        try:
            t, q_res3 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res3:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res3['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []
        
        gp_list = get_fixed_path_gp_two_hops(
            q_res1,
            q_res2,
            q_res3,
            gp_,
            richtung,
            gp_.matching_node_pairs,
            a,
            b,
            c,
            d,
            e
        )
    if richtung == 3:
        gp1 = GraphPattern([(SOURCE_VAR, a, b)])
        gp2 = GraphPattern([(d, c, b)])
        gp3 = GraphPattern([(d, e, TARGET_VAR)])
        
        values_s = {
            (SOURCE_VAR, ): [(tup[0], ) for tup in gp_.matching_node_pairs]
        }
        q = gp1.to_sparql_filter_by_count_in_out_query(
            values=values_s, count_node=b, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_next)
        logger.debug(q)
        try:
            t, q_res1 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res1:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res1['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []
        
        values_b = get_values([b], q_res1)
        q = gp2.to_sparql_filter_by_count_in_out_query(
            values=values_b, count_node=d, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_startpoint_two_sided)
        logger.debug(q)
        try:
            t, q_res2 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res2:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res2['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []
        
        values_t = {
            (TARGET_VAR, ): [(tup[1], ) for tup in gp_.matching_node_pairs]
        }
        q = gp3.to_sparql_filter_by_count_in_out_query(
            values=values_t, count_node=d, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_startpoint_two_sided)
        logger.debug(q)
        try:
            t, q_res3 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res3:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res3['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []
        
        gp_list = get_fixed_path_gp_two_hops(
            q_res1,
            q_res2,
            q_res3,
            gp_,
            richtung,
            gp_.matching_node_pairs,
            a,
            b,
            c,
            d,
            e
        )
    if richtung == 4:
        gp1 = GraphPattern([(SOURCE_VAR, a, b)])
        gp2 = GraphPattern([(d, c, b)])
        gp3 = GraphPattern([(TARGET_VAR, e, d)])
        
        values_s = {
            (SOURCE_VAR, ): [(tup[0], ) for tup in gp_.matching_node_pairs]
        }
        q = gp1.to_sparql_filter_by_count_in_out_query(
            values=values_s, count_node=b, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_endpoint_two_sided)
        logger.debug(q)
        try:
            t, q_res1 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res1:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res1['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []
        
        values_t = {
            (TARGET_VAR, ): [(tup[1], ) for tup in gp_.matching_node_pairs]
        }
        q = gp3.to_sparql_filter_by_count_in_out_query(
            values=values_t, count_node=d, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_next)
        logger.debug(q)
        try:
            t, q_res3 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res3:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res3['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []
        
        values_d = get_values([d], q_res3)
        q = gp2.to_sparql_filter_by_count_in_out_query(
            values=values_d, count_node=b, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_endpoint_two_sided)
        logger.debug(q)
        try:
            t, q_res2 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res2:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res2['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []
        
        gp_list = get_fixed_path_gp_two_hops(
            q_res1,
            q_res2,
            q_res3,
            gp_,
            richtung,
            gp_.matching_node_pairs,
            a,
            b,
            c,
            d,
            e
        )
    if richtung == 5:
        gp1 = GraphPattern([(b, a, SOURCE_VAR)])
        gp2 = GraphPattern([(b, c, d)])
        gp3 = GraphPattern([(d, e, TARGET_VAR)])

        values_s = {(SOURCE_VAR, ): [(tup[0], ) for tup in gp_.matching_node_pairs]}
        q = gp1.to_sparql_filter_by_count_in_out_query(
            values=values_s, count_node=b, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_subject_next)
        logger.debug(q)
        try:
            t, q_res1 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res1:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res1['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []

        values_b = get_values([b], q_res1)
        q = gp2.to_sparql_filter_by_count_in_out_query(
            values=values_b, count_node=d, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_next)
        logger.debug(q)
        try:
            t, q_res2 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res2:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res2['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []

        values_d = get_values([d], q_res2)
        q = gp3.to_sparql_select_sample_query(
            values=values_d, limit=limit_choose_endpoint
        )
        logger.debug(q)
        try:
            t, q_res3 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res3:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res3['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []

        gp_list = get_fixed_path_gp_two_hops(
            q_res1,
            q_res2,
            q_res3,
            gp_,
            richtung,
            gp_.matching_node_pairs,
            a,
            b,
            c,
            d,
            e
        )
    if richtung == 6:
        gp1 = GraphPattern([(b, a, SOURCE_VAR)])
        gp2 = GraphPattern([(b, c, d)])
        gp3 = GraphPattern([(TARGET_VAR, e, d)])

        values_t = {
            (TARGET_VAR, ): [(tup[1], ) for tup in gp_.matching_node_pairs]
        }
        q = gp3.to_sparql_filter_by_count_in_out_query(
            values=values_t, count_node=d, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_next)
        logger.debug(q)
        try:
            t, q_res3 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res3:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res3['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []

        values_s = {
            (SOURCE_VAR, ): [(tup[0], ) for tup in gp_.matching_node_pairs]
        }
        q = gp1.to_sparql_filter_by_count_in_out_query(
            values=values_s, count_node=b, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_startpoint_two_sided)
        logger.debug(q)
        try:
            t, q_res1 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res1:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res1['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []

        values_d = get_values([d], q_res3)
        q = gp2.to_sparql_filter_by_count_in_out_query(
            values=values_d, count_node=b, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_startpoint_two_sided)
        logger.debug(q)
        try:
            t, q_res2 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res2:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res2['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []

        gp_list = get_fixed_path_gp_two_hops(
            q_res1,
            q_res2,
            q_res3,
            gp_,
            richtung,
            gp_.matching_node_pairs,
            a,
            b,
            c,
            d,
            e
        )
    if richtung == 7:
        gp1 = GraphPattern([(b, a, SOURCE_VAR)])
        gp2 = GraphPattern([(d, c, b)])
        gp3 = GraphPattern([(d, e, TARGET_VAR)])

        values_t = {
            (TARGET_VAR, ): [(tup[1], ) for tup in gp_.matching_node_pairs]
        }
        q = gp3.to_sparql_filter_by_count_in_out_query(
            values=values_t, count_node=d, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_subject_next)
        logger.debug(q)
        try:
            t, q_res3 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res3:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res3['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []

        values_d = get_values([d], q_res3)
        q = gp2.to_sparql_filter_by_count_in_out_query(
            values=values_d, count_node=b, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_next)
        logger.debug(q)
        try:
            t, q_res2 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res2:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res2['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []

        values_b = get_values([b], q_res2)
        q = gp1.to_sparql_select_sample_query(
            values=values_b, limit=limit_choose_endpoint)
        logger.debug(q)
        try:
            t, q_res1 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res1:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res1['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []

        gp_list = get_fixed_path_gp_two_hops(
            q_res1,
            q_res2,
            q_res3,
            gp_,
            richtung,
            gp_.matching_node_pairs,
            a,
            b,
            c,
            d,
            e
        )
    if richtung == 8:
        gp1 = GraphPattern([(b, a, SOURCE_VAR)])
        gp2 = GraphPattern([(d, c, b)])
        gp3 = GraphPattern([(TARGET_VAR, e, d)])

        values_t = {
            (TARGET_VAR, ): [(tup[1], ) for tup in gp_.matching_node_pairs]
        }
        q = gp3.to_sparql_filter_by_count_in_out_query(
            values=values_t, count_node=d, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_next)
        logger.debug(q)
        try:
            t, q_res3 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res3:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res3['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []

        values_d = get_values([d], q_res3)
        q = gp2.to_sparql_filter_by_count_in_out_query(
            values=values_d, count_node=b, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_next)
        logger.debug(q)
        try:
            t, q_res2 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res2:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res2['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []

        values_b = get_values([b], q_res2)
        q = gp1.to_sparql_select_sample_query(
            values=values_b, limit=limit_choose_endpoint)
        logger.debug(q)
        try:
            t, q_res1 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res1:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res1['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []

        gp_list = get_fixed_path_gp_two_hops(
            q_res1,
            q_res2,
            q_res3,
            gp_,
            richtung,
            gp_.matching_node_pairs,
            a,
            b,
            c,
            d,
            e
        )
    
    return gp_list


# fixed den ein-hop-pfad zwischen Source und Target, fügt ihn dem Pattern hinzu
# und gibt die Liste der resultierenden Pattern zurück
# TODO nicht so sehr auf source a b. b c Target fokussieren.
def get_fixed_path_gp_three_hops(
        q_res1,
        q_res2,
        q_res3,
        q_res4,
        gp_main,
        richtung,
        stp,
        a,
        b,
        c,
        d,
        e,
        f,
        g
):
    # TODO: überlegen nicht nur verschieden Pattern für verschiedene Richtungen
    # zu machen, sondern auch in den Unterschiedlichen Ergebnissn anfangen
    # (Idee wäre z.B. die a bis e durch nummerierte random vars zu ersetzen und
    # sich dann zu überlegen wie man das übergibt, ob mans iwie immer entlang
    # des patterns schafft oder eher nicht.
    gp_list = []
    res_rows_path = ['results', 'bindings']
    bind1 = sparql_json_result_bindings_to_rdflib(
        get_path(q_res1, res_rows_path, default=[])
    )
    bind2 = sparql_json_result_bindings_to_rdflib(
        get_path(q_res2, res_rows_path, default=[])
    )
    bind3 = sparql_json_result_bindings_to_rdflib(
        get_path(q_res3, res_rows_path, default=[])
    )
    bind4 = sparql_json_result_bindings_to_rdflib(
        get_path(q_res4, res_rows_path, default=[])
    )
    for gtp in stp:
        for row4 in bind4:
            if gtp[1] == get_path(row4, [TARGET_VAR]):
                for row3 in bind3:
                    if get_path(row3, [f]) == get_path(row4, [f]):
                        for row2 in bind2:
                            if get_path(row2, [d]) == get_path(row3, [d]):
                                for row1 in bind1:
                                    if get_path(row1, [b]) == \
                                            get_path(row2, [b]) and \
                                            get_path(row1, [SOURCE_VAR]) == \
                                            gtp[0]:
                                        if richtung == 1:
                                            gp_ = GraphPattern([
                                                (SOURCE_VAR, get_path(row1, [a]), b),
                                                (b, get_path(row2, [c]), d),
                                                (d, get_path(row3, [e]), f),
                                                (f, get_path(row4, [g]), TARGET_VAR)
                                            ])
                                        elif richtung == 2:
                                            gp_ = GraphPattern([
                                                (SOURCE_VAR, get_path(row1, [a]), b),
                                                (b, get_path(row2, [c]), d),
                                                (d, get_path(row3, [e]), f),
                                                (TARGET_VAR, get_path(row4, [g]), f)
                                            ])
                                        else:  # dummy else, damit gp_ zugewiesen
                                            gp_ = GraphPattern([])
                                        gp_ = GraphPattern(chain(gp_, gp_main))
                                        if gp_ not in gp_list:
                                            gp_list.append(gp_)
                                        logger.debug(gtp)
    return gp_list


# add two hops.
def mutate_deep_narrow_three_hops(
        gp_, max_out=None, max_in=None, in_out=None, richtung=None
):
    vars_ = gp_.vars_in_graph
    if not (SOURCE_VAR in vars_ and TARGET_VAR in vars_):
        logger.debug('SOURCE or TARGET are not in gp: %s' % gp_)
        return []
    if not gp_.matching_node_pairs:
        logger.debug(
            'No matching node pairs, cant get better through adding constraints'
        )
        return []
    a = Variable('a')
    b = Variable('b')
    c = Variable('c')
    d = Variable('d')
    e = Variable('e')
    f = Variable('f')
    g = Variable('g')
    if richtung not in range(1, 17):
        richtung = random.choice(range(1, 17))
        logger.debug('Richtung %s wurde gewaehlt' % richtung)
    if richtung == 1:
        gp1 = GraphPattern([(SOURCE_VAR, a, b)])
        gp2 = GraphPattern([(b, c, d)])
        gp3 = GraphPattern([(d, e, f)])
        gp4 = GraphPattern([(f, g, TARGET_VAR)])

        values_s = {
            (SOURCE_VAR, ): [(tup[0], ) for tup in gp_.matching_node_pairs]
        }
        q = gp1.to_sparql_filter_by_count_in_out_query(
            values=values_s, count_node=b, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_next)
        logger.debug(q)
        try:
            t, q_res1 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res1:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res1['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []

        values_b = get_values([b], q_res1)
        q = gp2.to_sparql_filter_by_count_in_out_query(
            values=values_b, count_node=d, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_next)
        logger.debug(q)
        try:
            t, q_res2 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res2:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res2['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []

        values_d = get_values([d], q_res2)
        q = gp3.to_sparql_filter_by_count_in_out_query(
            values=values_d, count_node=f, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_next)
        logger.debug(q)
        try:
            t, q_res3 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res3:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res3['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []

        values_f = get_values([f], q_res3)
        q = gp4.to_sparql_select_sample_query(
            values=values_f, limit=limit_choose_endpoint
        )
        logger.debug(q)
        try:
            t, q_res4 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res4:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res4['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []

        gp_list = get_fixed_path_gp_three_hops(
            q_res1,
            q_res2,
            q_res3,
            q_res4,
            gp_,
            richtung,
            gp_.matching_node_pairs,
            a,
            b,
            c,
            d,
            e,
            f,
            g
        )
    elif richtung == 2:
        gp1 = GraphPattern([(SOURCE_VAR, a, b)])
        gp2 = GraphPattern([(b, c, d)])
        gp3 = GraphPattern([(d, e, f)])
        gp4 = GraphPattern([(TARGET_VAR, g, f)])

        values_s = {
            (SOURCE_VAR, ): [(tup[0], ) for tup in gp_.matching_node_pairs]
        }
        q = gp1.to_sparql_filter_by_count_in_out_query(
            values=values_s, count_node=b, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_next)
        logger.debug(q)
        try:
            t, q_res1 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res1:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res1['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []

        values_b = get_values([b], q_res1)
        q = gp2.to_sparql_filter_by_count_in_out_query(
            values=values_b, count_node=d, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_next)
        logger.debug(q)
        try:
            t, q_res2 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res2:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res2['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []

        values_d = get_values([d], q_res2)
        q = gp3.to_sparql_filter_by_count_in_out_query(
            values=values_d, count_node=f, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_endpoint_two_sided)
        logger.debug(q)
        try:
            t, q_res3 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res3:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res3['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []

        values_t = {
            (TARGET_VAR,): [(tup[1],) for tup in gp_.matching_node_pairs]
        }
        q = gp4.to_sparql_filter_by_count_in_out_query(
            values=values_t, count_node=f, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_endpoint_two_sided)
        logger.debug(q)
        try:
            t, q_res4 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res4:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res4['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []

        gp_list = get_fixed_path_gp_three_hops(
            q_res1,
            q_res2,
            q_res3,
            q_res4,
            gp_,
            richtung,
            gp_.matching_node_pairs,
            a,
            b,
            c,
            d,
            e,
            f,
            g
        )
    elif richtung == 3:
        gp1 = GraphPattern([(SOURCE_VAR, a, b)])
        gp2 = GraphPattern([(b, c, d)])
        gp3 = GraphPattern([(f, e, d)])
        gp4 = GraphPattern([(f, g, TARGET_VAR)])

        values_s = {
            (SOURCE_VAR, ): [(tup[0], ) for tup in gp_.matching_node_pairs]
        }
        q = gp1.to_sparql_filter_by_count_in_out_query(
            values=values_s, count_node=b, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_next)
        logger.debug(q)
        try:
            t, q_res1 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res1:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res1['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []

        values_b = get_values([b], q_res1)
        q = gp2.to_sparql_filter_by_count_in_out_query(
            values=values_b, count_node=d, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_next)
        logger.debug(q)
        try:
            t, q_res2 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res2:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res2['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []

        values_d = get_values([d], q_res2)
        q = gp3.to_sparql_filter_by_count_in_out_query(
            values=values_d, count_node=f, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_startpoint_two_sided)
        logger.debug(q)
        try:
            t, q_res3 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res3:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res3['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []

        values_t = {
            (TARGET_VAR,): [(tup[1],) for tup in gp_.matching_node_pairs]
        }
        q = gp4.to_sparql_filter_by_count_in_out_query(
            values=values_t, count_node=f, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_startpoint_two_sided)
        logger.debug(q)
        try:
            t, q_res4 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res4:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res4['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []

        gp_list = get_fixed_path_gp_three_hops(
            q_res1,
            q_res2,
            q_res3,
            q_res4,
            gp_,
            richtung,
            gp_.matching_node_pairs,
            a,
            b,
            c,
            d,
            e,
            f,
            g
        )
    elif richtung == 4:
        gp1 = GraphPattern([(SOURCE_VAR, a, b)])
        gp2 = GraphPattern([(b, c, d)])
        gp3 = GraphPattern([(f, e, d)])
        gp4 = GraphPattern([(TARGET_VAR, g, f)])

        values_s = {
            (SOURCE_VAR, ): [(tup[0], ) for tup in gp_.matching_node_pairs]
        }
        q = gp1.to_sparql_filter_by_count_in_out_query(
            values=values_s, count_node=b, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_next)
        logger.debug(q)
        try:
            t, q_res1 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res1:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res1['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []

        values_b = get_values([b], q_res1)
        q = gp2.to_sparql_filter_by_count_in_out_query(
            values=values_b, count_node=d, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_endpoint_two_sided)
        logger.debug(q)
        try:
            t, q_res2 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res2:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res2['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []

        values_t = {
            (TARGET_VAR,): [(tup[1],) for tup in gp_.matching_node_pairs]
        }
        q = gp4.to_sparql_filter_by_count_in_out_query(
            values=values_t, count_node=f, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_next)
        logger.debug(q)
        try:
            t, q_res4 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res4:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res4['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []

        values_f = get_values([f], q_res4)
        q = gp3.to_sparql_filter_by_count_in_out_query(
            values=values_f, count_node=d, in_out=in_out, max_out=max_out,
            max_in=max_in, limit=limit_endpoint_two_sided)
        logger.debug(q)
        try:
            t, q_res3 = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not q_res3:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not q_res3['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []

        gp_list = get_fixed_path_gp_three_hops(
            q_res1,
            q_res2,
            q_res3,
            q_res4,
            gp_,
            richtung,
            gp_.matching_node_pairs,
            a,
            b,
            c,
            d,
            e,
            f,
            g
        )

    return gp_list


def get_fixed_path_gp_n_hops(
        res_q, gp_, n, direct, stp, node, hn_ind, hop
):
    gp_list = []
    res_rows_path = ['results', 'bindings']
    bind = []
    for res_q_i in res_q:
        bind.append(sparql_json_result_bindings_to_rdflib(
            get_path(res_q_i, res_rows_path, default=[]))
        )
    hit_paths = []
    hit_paths_help = []

    if hn_ind == 0:
        for row in bind[0]:
            for mnp in stp:
                if mnp[0] == (get_path(row, [node[0]])):
                    hit_paths.append([[
                        mnp[0],
                        get_path(row, [hop[0]]),
                        get_path(row, [node[1]])
                    ]])
        for i in range(1, n+1):
            for path in hit_paths:
                for row in bind[i]:
                    if path[i-1][2] == get_path(row, [node[i]]):
                        path_h = path + [[
                            path[i-1][2],
                            get_path(row, [hop[i]]),
                            get_path(row, [node[i+1]])
                        ]]
                        hit_paths_help.append(path_h)
            hit_paths = hit_paths_help
            hit_paths_help = []

    elif hn_ind == n+1:
        for row in bind[n]:
            for mnp in stp:
                if mnp[1] == (get_path(row, [node[n+1]])):
                    hit_paths.append([[
                        get_path(row, [node[n]]),
                        get_path(row, [hop[n]]),
                        mnp[1]
                    ]])
        for i in range(n-1, -1, -1):
            for path in hit_paths:
                for row in bind[i]:
                    if path[(n-1)-i][0] == get_path(row, [node[i+1]]):
                        path_h = path.append([[
                            get_path(row, [node[i]],
                                     get_path(row, [hop[i]]),
                                     path[(n-1)-i][0])
                        ]])
                        hit_paths_help.append(path_h)
        hit_paths = hit_paths_help
        hit_paths_help = []
        for path in hit_paths:
            path.reverse()

    else:
        hit_paths_l = []
        hit_paths_r = []
        # get the hits of hit_node to start from
        for row_l in bind[hn_ind-1]:
            for row_r in bind[hn_ind]:
                if get_path(row_l, [node[hn_ind]]) == \
                        get_path(row_r, [node[hn_ind]]):
                    hit_paths_l.append([[
                        get_path(row_l, [node[hn_ind-1]]),
                        get_path(row_l, [hop[hn_ind-1]]),
                        get_path(row_l, [node[hn_ind]])
                    ]])
                    hit_paths_r.append([[
                        get_path(row_r, [node[hn_ind]]),
                        get_path(row_r, [hop[hn_ind]]),
                        get_path(row_r, [node[hn_ind+1]])
                    ]])
        # get the path from hit node to targets
        for i in range(hn_ind+1, n+1):
            for path in hit_paths_r:
                for row in bind[i]:
                    if path[i-(hn_ind+1)][2] == get_path(row, [node[i]]):
                        path_h = path + [[
                            path[i-(hn_ind+1)][2],
                            get_path(row, [hop[i]]),
                            get_path(row, [node[i+1]])
                        ]]
                        hit_paths_help.append(path_h)
            hit_paths_r = hit_paths_help
            hit_paths_help = []
        # get the path from hit node to sources
        for i in range(hn_ind, -1, -1):
            for path in hit_paths_l:
                for row in bind[i]:
                    if path[hn_ind-i][0] == get_path(row, [node[i+1]]):
                        path_h = path + [[
                            get_path(row, [node[i]]),
                            get_path(row, [hop[i]]),
                            path[hn_ind-i][0]
                        ]]
                        hit_paths_help.append(path_h)
            hit_paths_l = hit_paths_help
            hit_paths_help = []
        # get the full path from source to target
        for path_l in hit_paths_l:
            path_l.reverse()
            for path_r in hit_paths_r:
                if path_l[hn_ind][2] == path_r[0][0]:
                    hit_paths.append(path_l + path_r)
        # filter the paths, over stp-hits

    hit_paths = filter_stp_hits(hit_paths, stp)

    # Make Graph_Pattern_with fixed hops out of the found paths
    for path in hit_paths:
        gp_list.append(
            GraphPattern(
                chain(
                    GraphPattern([
                        (node[i], path[i][1], node[i+1]) if direct(i) == 1
                        else (node[i+1], path[i][1], node[i])
                        for i in range(n+1)
                    ]),
                    gp_
                )
            )
        )

    return gp_list


def filter_stp_hits(
        hit_paths, stp
):
    res = []
    for hit in hit_paths:
        for mnp in stp:
            if (mnp[0] == hit[0][0]) and (mnp[1] == hit[len(hit)-1][2]):
                res.append(hit)
    return res


def mutate_deep_narrow_n_hops(
        gp_, n, max_out=None, max_in=None, in_out=None, direct=None
):
    vars_ = gp_.vars_in_graph
    if SOURCE_VAR not in vars_ and TARGET_VAR not in vars_:
        logger.info('SOURCE or TARGET are not in gp: %s' % gp_)
        return []
    if not gp_.matching_node_pairs:
        logger.info(
            'No matching node pairs, cant get better through adding constraints'
        )
        return []
    if n < 1:
        logger.info('Cannot add less than one hop')
        return []
    # setting up lists for nodes, hops, values, gp_helpers, query-results
    node = [SOURCE_VAR]
    for i in range(n):
        node.append(gen_random_var())
    node.append(TARGET_VAR)
    hop = []
    for i in range(n+1):
        hop.append(gen_random_var())
    if direct is None or len(direct) != n+1:
        logger.info('No direction chosen, or direction tuple with false length')
        direct = []
        for i in range(n+1):
            direct.append(0)
    gp_helper = []
    for i in range(n+1):
        if direct[i] == 0:
            direct[i] = random.choice([-1, 1])
        if direct[i] == 1:
            gp_helper.append(
                GraphPattern([(node[i], hop[i], node[i + 1])])
            )
        else:
            gp_helper.append(
                GraphPattern([(node[i + 1], hop[i], node[i])])
            )
    values = []
    for i in range(n+2):
        values.append({})
    values[0] = {
        (SOURCE_VAR, ): [(tup[0], ) for tup in gp_.matching_node_pairs]
    }
    values[n+1] = {
        (TARGET_VAR, ): [(tup[1], ) for tup in gp_.matching_node_pairs]
    }
    res_q = []
    for i in range(n+1):
        res_q.append({})

    # selecting an random "hit_node" => Node to check the random hits
    hit_node = random.choice(node)
    hn_ind = node.index(hit_node)

    # TODO: use direct for cases in queriing
    # Querieing
    # From source to target if hit_node is target:
    if hit_node == TARGET_VAR:
        # Firing the queries for the first n-2 steps
        for i in range(0, n):
            if gp_helper[i][0][0] == node[i]:
                q = gp_helper[i].to_sparql_filter_by_count_in_out_query(
                    values=values[i], count_node=node[i+1], in_out=in_out,
                    max_out=max_out, max_in=max_in, limit=limit_next)
            else:
                q = gp_helper[i].to_sparql_filter_by_count_in_out_query(
                    values=values[i], count_node=node[i+1], in_out=in_out,
                    max_out=max_out, max_in=max_in, limit=limit_subject_next)
            logger.info(q)
            try:
                t, res_q[i] = run_query(q)
            except:
                logger.info('Die Query (s.o.) hat nicht geklappt')
                return []
            if not res_q[i]:
                logger.info('Die Query (s.o.) hat kein Ergebnis geliefert')
                return []
            elif not res_q[i]['results']['bindings']:
                logger.info('Die Query (s.o.) hat keine gebundenen Variablen')
                return []
            values[i+1] = get_values([node[i+1]], res_q[i])
        # Firing the last query for the target hits:
        if gp_helper[n][0][0] == node[n-1]:
            q = gp_helper[n].to_sparql_select_sample_query(
                values=values[n], limit=limit_choose_endpoint)
        else:
            q = gp_helper[n].to_sparql_select_sample_query(
                values=values[n], limit=limit_choose_subject_endpoint)
        logger.info(q)
        try:
            t, res_q[n] = run_query(q)
        except:
            logger.info('Die Query (s.o.) hat nicht geklappt')
            return []
        if not res_q[n]:
            logger.info('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not res_q[n]['results']['bindings']:
            logger.info('Die Query (s.o.) hat keine gebundenen Variablen')
            return []

    # From target to source if hit_node is source:
    elif hit_node == SOURCE_VAR:
        # Firing the queries for the first n-2 steps
        for i in range(n, 0, -1):
            if gp_helper[i][0][0] == node[i+1]:
                q = gp_helper[i].to_sparql_filter_by_count_in_out_query(
                    values=values[i+1], count_node=node[i], in_out=in_out,
                    max_out=max_out, max_in=max_in, limit=limit_next)
            else:
                q = gp_helper[i].to_sparql_filter_by_count_in_out_query(
                    values=values[i+1], count_node=node[i], in_out=in_out,
                    max_out=max_out, max_in=max_in, limit=limit_subject_next)
            logger.info(q)
            try:
                t, res_q[i] = run_query(q)
            except:
                logger.info('Die Query (s.o.) hat nicht geklappt')
                return []
            if not res_q[i]:
                logger.info('Die Query (s.o.) hat kein Ergebnis geliefert')
                return []
            elif not res_q[i]['results']['bindings']:
                logger.info('Die Query (s.o.) hat keine gebundenen Variablen')
                return []
            values[i] = get_values([node[i]], res_q[i])
        # Firing the last query for the target hits:
        if gp_helper[0][0][0] == node[1]:
            q = gp_helper[0].to_sparql_select_sample_query(
                values=values[1], limit=limit_choose_endpoint)
        else:
            q = gp_helper[0].to_sparql_select_sample_query(
                values=values[1], limit=limit_choose_subject_endpoint)
        logger.info(q)
        try:
            t, res_q[0] = run_query(q)
        except:
            logger.info('Die Query (s.o.) hat nicht geklappt')
            return []
        if not res_q[0]:
            logger.info('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not res_q[0]['results']['bindings']:
            logger.info('Die Query (s.o.) hat keine gebundenen Variablen')
            return []

    # From both sides to the hit_node:
    else:
        # firing the queries from source to the last node before hit_node
        for i in range(0, hn_ind-1):
            if gp_helper[i][0][0] == node[i]:
                q = gp_helper[i].to_sparql_filter_by_count_in_out_query(
                    values=values[i], count_node=node[i+1], in_out=in_out,
                    max_out=max_out, max_in=max_in, limit=limit_next)
            else:
                q = gp_helper[i].to_sparql_filter_by_count_in_out_query(
                    values=values[i], count_node=node[i+1], in_out=in_out,
                    max_out=max_out, max_in=max_in, limit=limit_subject_next)
            logger.info(q)
            try:
                t, res_q[i] = run_query(q)
            except:
                logger.info('Die Query (s.o.) hat nicht geklappt')
                return []
            if not res_q[i]:
                logger.info('Die Query (s.o.) hat kein Ergebnis geliefert')
                return []
            elif not res_q[i]['results']['bindings']:
                logger.info('Die Query (s.o.) hat keine gebundenen Variablen')
                return []
            values[i+1] = get_values([node[i+1]], res_q[i])
            # Firing the queries from target to the last node before hit node
        for i in range(n, hn_ind, -1):
            if gp_helper[i][0][0] == node[i+1]:
                q = gp_helper[i].to_sparql_filter_by_count_in_out_query(
                    values=values[i+1], count_node=node[i], in_out=in_out,
                    max_out=max_out, max_in=max_in, limit=limit_next)
            else:
                q = gp_helper[i].to_sparql_filter_by_count_in_out_query(
                    values=values[i+1], count_node=node[i], in_out=in_out,
                    max_out=max_out, max_in=max_in, limit=limit_subject_next)
            logger.info(q)
            try:
                t, res_q[i] = run_query(q)
            except:
                logger.info('Die Query (s.o.) hat nicht geklappt')
                return []
            if not res_q[i]:
                logger.info('Die Query (s.o.) hat kein Ergebnis geliefert')
                return []
            elif not res_q[i]['results']['bindings']:
                logger.info('Die Query (s.o.) hat keine gebundenen Variablen')
                return []
            values[i] = get_values([node[i]], res_q[i])
        # feuere die letzten beiden queries richtung hit_node ab.
        # Dabei unterscheide nach Richtungen beider queries.
        if ((gp_helper[hn_ind-1][0][0] == node[hn_ind-1]) and   # hit is Object
                (gp_helper[hn_ind][0][0] == node[hn_ind+1])):   # hit is Object
            q = gp_helper[hn_ind-1].to_sparql_filter_by_count_in_out_query(
                values=values[hn_ind-1], count_node=node[hn_ind], in_out=in_out,
                max_out=max_out, max_in=max_in, limit=limit_endpoint_two_sided)
            logger.info(q)
            try:
                t, res_q[hn_ind-1] = run_query(q)
            except:
                logger.info('Die Query (s.o.) hat nicht geklappt')
                return []
            if not res_q[hn_ind-1]:
                logger.info('Die Query (s.o.) hat kein Ergebnis geliefert')
                return []
            elif not res_q[hn_ind-1]['results']['bindings']:
                logger.info('Die Query (s.o.) hat keine gebundenen Variablen')
                return []
            q = gp_helper[hn_ind].to_sparql_filter_by_count_in_out_query(
                values=values[hn_ind+1], count_node=node[hn_ind], in_out=in_out,
                max_out=max_out, max_in=max_in, limit=limit_endpoint_two_sided)
            logger.info(q)
            try:
                t, res_q[hn_ind] = run_query(q)
            except:
                logger.info('Die Query (s.o.) hat nicht geklappt')
                return []
            if not res_q[hn_ind]:
                logger.info('Die Query (s.o.) hat kein Ergebnis geliefert')
                return []
            elif not res_q[hn_ind]['results']['bindings']:
                logger.info('Die Query (s.o.) hat keine gebundenen Variablen')
                return []
        elif ((gp_helper[hn_ind-1][0][0] == node[hn_ind]) and   # hit is Subject
              (gp_helper[hn_ind][0][0] == node[hn_ind])):       # hit is Subject
            q = gp_helper[hn_ind-1].to_sparql_filter_by_count_in_out_query(
                values=values[hn_ind-1], count_node=node[hn_ind], in_out=in_out,
                max_out=max_out, max_in=max_in, limit=limit_startpoint_two_sided)
            logger.info(q)
            try:
                t, res_q[hn_ind-1] = run_query(q)
            except:
                logger.info('Die Query (s.o.) hat nicht geklappt')
                return []
            if not res_q[hn_ind-1]:
                logger.info('Die Query (s.o.) hat kein Ergebnis geliefert')
                return []
            elif not res_q[hn_ind-1]['results']['bindings']:
                logger.info('Die Query (s.o.) hat keine gebundenen Variablen')
                return []
            q = gp_helper[hn_ind].to_sparql_filter_by_count_in_out_query(
                values=values[hn_ind+1], count_node=node[hn_ind], in_out=in_out,
                max_out=max_out, max_in=max_in, limit=limit_startpoint_two_sided)
            logger.info(q)
            try:
                t, res_q[hn_ind] = run_query(q)
            except:
                logger.info('Die Query (s.o.) hat nicht geklappt')
                return []
            if not res_q[hn_ind]:
                logger.info('Die Query (s.o.) hat kein Ergebnis geliefert')
                return []
            elif not res_q[hn_ind]['results']['bindings']:
                logger.info('Die Query (s.o.) hat keine gebundenen Variablen')
                return []
        elif ((gp_helper[hn_ind-1][0][0] == node[hn_ind-1]) and  # hit is Object
                (gp_helper[hn_ind][0][0] == node[hn_ind])):      # hit is Subject
            q = gp_helper[hn_ind-1].to_sparql_filter_by_count_in_out_query(
                values=values[hn_ind-1], count_node=node[hn_ind], in_out=in_out,
                max_out=max_out, max_in=max_in, limit=limit_obj_to_subj)
            logger.info(q)
            try:
                t, res_q[hn_ind-1] = run_query(q)
            except:
                logger.info('Die Query (s.o.) hat nicht geklappt')
                return []
            if not res_q[hn_ind-1]:
                logger.info('Die Query (s.o.) hat kein Ergebnis geliefert')
                return []
            elif not res_q[hn_ind-1]['results']['bindings']:
                logger.info('Die Query (s.o.) hat keine gebundenen Variablen')
                return []
            q = gp_helper[hn_ind].to_sparql_filter_by_count_in_out_query(
                values=values[hn_ind+1], count_node=node[hn_ind], in_out=in_out,
                max_out=max_out, max_in=max_in, limit=limit_subj_to_obj)
            logger.info(q)
            try:
                t, res_q[hn_ind] = run_query(q)
            except:
                logger.info('Die Query (s.o.) hat nicht geklappt')
                return []
            if not res_q[hn_ind]:
                logger.info('Die Query (s.o.) hat kein Ergebnis geliefert')
                return []
            elif not res_q[hn_ind]['results']['bindings']:
                logger.info('Die Query (s.o.) hat keine gebundenen Variablen')
                return []
        elif ((gp_helper[hn_ind-1][0][0] == node[hn_ind]) and   # hit is Subject
                (gp_helper[hn_ind][0][0] == node[hn_ind+1])):   # hit is Object
            q = gp_helper[hn_ind-1].to_sparql_filter_by_count_in_out_query(
                values=values[hn_ind-1], count_node=node[hn_ind], in_out=in_out,
                max_out=max_out, max_in=max_in, limit=limit_subj_to_obj)
            logger.info(q)
            try:
                t, res_q[hn_ind-1] = run_query(q)
            except:
                logger.info('Die Query (s.o.) hat nicht geklappt')
                return []
            if not res_q[hn_ind-1]:
                logger.info('Die Query (s.o.) hat kein Ergebnis geliefert')
                return []
            elif not res_q[hn_ind-1]['results']['bindings']:
                logger.info('Die Query (s.o.) hat keine gebundenen Variablen')
                return []
            q = gp_helper[hn_ind].to_sparql_filter_by_count_in_out_query(
                values=values[hn_ind+1], count_node=node[hn_ind], in_out=in_out,
                max_out=max_out, max_in=max_in, limit=limit_obj_to_subj)
            logger.info(q)
            try:
                t, res_q[hn_ind] = run_query(q)
            except:
                logger.info('Die Query (s.o.) hat nicht geklappt')
                return []
            if not res_q[hn_ind]:
                logger.info('Die Query (s.o.) hat kein Ergebnis geliefert')
                return []
            elif not res_q[hn_ind]['results']['bindings']:
                logger.info('Die Query (s.o.) hat keine gebundenen Variablen')
                return []

    gp_list = get_fixed_path_gp_n_hops(
        res_q, gp_, n, direct, gp_.matching_node_pairs, node, hn_ind, hop
    )

    return gp_list


# erste Version, komplett straight forward
def mutate_deep_narrow_1(
        gp_, gtps, n, direct=None, gp_in=False
):
    node = [SOURCE_VAR]
    for i in range(n):
        node.append(Variable('n%i' % i))
    node.append(TARGET_VAR)
    hop = []
    for i in range(n + 1):
        hop.append(Variable('p%i' % i))
    if direct is None or len(direct) != n + 1:
        logger.debug(
            'No direction chosen, or direction tuple with false length'
        )
        direct = []
        for i in range(n + 1):
            direct.append(0)
    gp_helper = []
    for i in range(n + 1):
        if direct[i] == 0:
            direct[i] = random.choice([-1, 1])
        if direct[i] == 1:
            gp_helper.append(
                GraphPattern([(node[i], hop[i], node[i + 1])])
            )
        else:
            gp_helper.append(
                GraphPattern([(node[i + 1], hop[i], node[i])])
            )
    values = {}
    values[SOURCE_VAR] = {(SOURCE_VAR,): [(tup[0],) for tup in gtps]}
    values[TARGET_VAR] = {(TARGET_VAR,): [(tup[1],) for tup in gtps]}
    values['st'] = {(SOURCE_VAR, TARGET_VAR): gtps}
    res_q = []
    for i in range(n + 1):
        res_q.append({})

    # Queries für die Schritte
    valueblocks = {}
    valueblocks[SOURCE_VAR] = values[SOURCE_VAR]
    for i in range(n+1):
        q = gp_.to_sparql_deep_narrow_path_query(
            hop[i], node[i+1], valueblocks, gp_helper[:i+1], gp_in=gp_in
        )
        logger.debug(q)
        try:
            t, res_q[i] = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not res_q[i]:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not res_q[i]['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []
        values[hop[i]] = get_values([hop[i]], res_q[i])
        valueblocks[hop[i]] = {
            (hop[i],): random.sample(
                values[hop[i]][(hop[i],)],
                min(10, len(values[hop[i]][(hop[i],)]))
            )
        }

    # Query fürs Ergebnis
    gp_help = GraphPattern([
                    (node[i], hop[i], node[i+1]) if direct[i] == 1
                    else (node[i+1], hop[i], node[i])
                    for i in range(n+1)
                    ])
    # gemeinsamer source/target-block, damit nur "richtige" Pfade gefunden
    # werden
    del valueblocks[SOURCE_VAR]
    valueblocks['st'] = values['st']
    q = gp_.to_sparql_deep_narrow_path_inst_query_old(hop, valueblocks, gp_help, gp_in=gp_in)
    logger.debug(q)
    try:
        t, res_q_inst = run_query(q)
    except:
        logger.debug('Die Query (s.o.) hat nicht geklappt')
        return []
    if not res_q_inst:
        logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
        return []
    elif not res_q_inst['results']['bindings']:
        logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
        return []
    res = []
    res_rows_path = ['results', 'bindings']
    bind = sparql_json_result_bindings_to_rdflib(
        get_path(res_q_inst, res_rows_path, default=[])
    )
    for row in bind:
        gp_res = GraphPattern([
            (node[i], get_path(row, [hop[i]]), node[i + 1]) if direct[i] == 1
            else (node[i + 1], get_path(row, [hop[i]]), node[i])
            for i in range(n + 1)
        ])
        res.append(gp_res)

    return res


# zweite Version: Query für letzten step bekommt schon die Targets
def mutate_deep_narrow_2(
        gp_, gtps, n, direct=None, gp_in=False
):
    node = [SOURCE_VAR]
    for i in range(n):
        node.append(Variable('n%i' % i))
    node.append(TARGET_VAR)
    hop = []
    for i in range(n + 1):
        hop.append(Variable('p%i' % i))
    if direct is None or len(direct) != n + 1:
        logger.debug(
            'No direction chosen, or direction tuple with false length'
        )
        direct = []
        for i in range(n + 1):
            direct.append(0)
    gp_helper = []
    for i in range(n + 1):
        if direct[i] == 0:
            direct[i] = random.choice([-1, 1])
        if direct[i] == 1:
            gp_helper.append(
                GraphPattern([(node[i], hop[i], node[i + 1])])
            )
        else:
            gp_helper.append(
                GraphPattern([(node[i + 1], hop[i], node[i])])
            )
    values = {}
    values[SOURCE_VAR] = {(SOURCE_VAR,): [(tup[0],) for tup in gtps]}
    values[TARGET_VAR] = {(TARGET_VAR,): [(tup[1],) for tup in gtps]}
    values['st'] = {(SOURCE_VAR, TARGET_VAR): gtps}
    res_q = []
    for i in range(n + 1):
        res_q.append({})

    # Queries für die Schritte
    valueblocks = {}
    valueblocks[SOURCE_VAR] = values[SOURCE_VAR]
    for i in range(n):
        q = gp_.to_sparql_deep_narrow_path_query(
            hop[i], node[i+1], valueblocks, gp_helper[:i+1], gp_in=gp_in
        )
        logger.debug(q)
        try:
            t, res_q[i] = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not res_q[i]:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not res_q[i]['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []
        values[hop[i]] = get_values([hop[i]], res_q[i])
        valueblocks[hop[i]] = {
            (hop[i],): random.sample(
                values[hop[i]][(hop[i],)],
                min(10, len(values[hop[i]][(hop[i],)]))
            )
        }

    # gemeinsamer source/target-block, damit nur "richtige" Pfade gefunden
    # werden
    del valueblocks[SOURCE_VAR]
    valueblocks['st'] = values['st']
    q = gp_.to_sparql_deep_narrow_path_inst_query(
        hop, valueblocks, gp_helper, gp_in=gp_in
    )
    logger.debug(q)
    try:
        t, res_q_inst = run_query(q)
    except:
        logger.debug('Die Query (s.o.) hat nicht geklappt')
        return []
    if not res_q_inst:
        logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
        return []
    elif not res_q_inst['results']['bindings']:
        logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
        return []
    res = []
    res_rows_path = ['results', 'bindings']
    bind = sparql_json_result_bindings_to_rdflib(
        get_path(res_q_inst, res_rows_path, default=[])
    )
    for row in bind:
        gp_res = GraphPattern([
            (node[i], get_path(row, [hop[i]]), node[i + 1]) if direct[i] == 1
            else (node[i + 1], get_path(row, [hop[i]]), node[i])
            for i in range(n + 1)
        ])
        res.append(gp_res)

    return res


# dritte Version: BIDI straight forward
def mutate_deep_narrow_3(
        gp_, gtps, n, direct=None, gp_in=False
):
    node = [SOURCE_VAR]
    for i in range(n):
        node.append(Variable('n%i' % i))
    node.append(TARGET_VAR)
    hop = []
    for i in range(n + 1):
        hop.append(Variable('p%i' % i))
    if direct is None or len(direct) != n + 1:
        logger.debug(
            'No direction chosen, or direction tuple with false length'
        )
        direct = []
        for i in range(n + 1):
            direct.append(0)
    gp_helper = []
    for i in range(n + 1):
        if direct[i] == 0:
            direct[i] = random.choice([-1, 1])
        if direct[i] == 1:
            gp_helper.append(
                GraphPattern([(node[i], hop[i], node[i + 1])])
            )
        else:
            gp_helper.append(
                GraphPattern([(node[i + 1], hop[i], node[i])])
            )
    values = {}
    values[SOURCE_VAR] = {(SOURCE_VAR,): [(tup[0],) for tup in gtps]}
    values[TARGET_VAR] = {(TARGET_VAR,): [(tup[1],) for tup in gtps]}
    values['st'] = {(SOURCE_VAR, TARGET_VAR): gtps}
    res_q = []
    for i in range(n+1):
        res_q.append({})

    # Queries für die Schritte
    valueblocks_s = {}
    valueblocks_s[SOURCE_VAR] = values[SOURCE_VAR]
    valueblocks_t = {}
    valueblocks_t[TARGET_VAR] = values[TARGET_VAR]
    for i in range(int((n / 2) + 1)):
        q = gp_.to_sparql_deep_narrow_path_query(
            hop[i], node[i+1], valueblocks_s, gp_helper[:i+1], gp_in=gp_in
        )
        logger.debug(q)
        try:
            t, res_q[i] = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not res_q[i]:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not res_q[i]['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []
        values[hop[i]] = get_values([hop[i]], res_q[i])
        valueblocks_s[hop[i]] = {
            (hop[i],): random.sample(
                values[hop[i]][(hop[i],)],
                min(10, len(values[hop[i]][(hop[i],)]))
            )
        }
        if n-i != i:
            q = gp_.to_sparql_deep_narrow_path_query(
                hop[n-i],
                node[n-i],
                valueblocks_t,
                gp_helper[n-i:],
                startvar=TARGET_VAR,
                gp_in=gp_in
            )
            logger.debug(q)
            try:
                t, res_q[n-i] = run_query(q)
            except:
                logger.debug('Die Query (s.o.) hat nicht geklappt')
                return []
            if not res_q[n-i]:
                logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
                return []
            elif not res_q[n-i]['results']['bindings']:
                logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
                return []
            values[hop[n-i]] = get_values([hop[n-i]], res_q[n-i])
            valueblocks_t[hop[n-i]] = {
                (hop[n-i],): random.sample(
                    values[hop[n-i]][(hop[n-i],)],
                    min(10, len(values[hop[n-i]][(hop[n-i],)]))
                )
            }

    # Query fürs Ergebnis
    gp_help = GraphPattern([
                    (node[i], hop[i], node[i+1]) if direct[i] == 1
                    else (node[i+1], hop[i], node[i])
                    for i in range(n+1)
                    ])
    # gemeinsamer source/target-block, damit nur "richtige" Pfade gefunden
    # werden
    valueblocks = {}
    for key in valueblocks_s:
        if key is not SOURCE_VAR:
            valueblocks[key] = valueblocks_s[key]
    for key in valueblocks_t:
        if key is not TARGET_VAR:
            valueblocks[key] = valueblocks_t[key]
    valueblocks['st'] = values['st']
    q = gp_.to_sparql_deep_narrow_path_inst_query_old(hop, valueblocks, gp_help, gp_in=gp_in)
    logger.debug(q)
    try:
        t, res_q_inst = run_query(q)
    except:
        logger.debug('Die Query (s.o.) hat nicht geklappt')
        return []
    if not res_q_inst:
        logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
        return []
    elif not res_q_inst['results']['bindings']:
        logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
        return []
    res = []
    res_rows_path = ['results', 'bindings']
    bind = sparql_json_result_bindings_to_rdflib(
        get_path(res_q_inst, res_rows_path, default=[])
    )
    for row in bind:
        gp_res = GraphPattern([
            (node[i], get_path(row, [hop[i]]), node[i + 1]) if direct[i] == 1
            else (node[i + 1], get_path(row, [hop[i]]), node[i])
            for i in range(n + 1)
        ])
        res.append(gp_res)

    return res


# vierte Version: BIDI with instantiation in last step
def mutate_deep_narrow_4(
        gp_, gtps, n, direct=None, gp_in=False
):
    node = [SOURCE_VAR]
    for i in range(n):
        node.append(Variable('n%i' % i))
    node.append(TARGET_VAR)
    hop = []
    for i in range(n + 1):
        hop.append(Variable('p%i' % i))
    if direct is None or len(direct) != n + 1:
        logger.debug(
            'No direction chosen, or direction tuple with false length'
        )
        direct = []
        for i in range(n + 1):
            direct.append(0)
    gp_helper = []
    for i in range(n + 1):
        if direct[i] == 0:
            direct[i] = random.choice([-1, 1])
        if direct[i] == 1:
            gp_helper.append(
                GraphPattern([(node[i], hop[i], node[i + 1])])
            )
        else:
            gp_helper.append(
                GraphPattern([(node[i + 1], hop[i], node[i])])
            )
    values = {}
    values[SOURCE_VAR] = {(SOURCE_VAR,): [(tup[0],) for tup in gtps]}
    values[TARGET_VAR] = {(TARGET_VAR,): [(tup[1],) for tup in gtps]}
    values['st'] = {(SOURCE_VAR, TARGET_VAR): gtps}
    res_q = []
    for i in range(n+1):
        res_q.append({})

    # Queries für die Schritte
    valueblocks_s = {}
    valueblocks_s[SOURCE_VAR] = values[SOURCE_VAR]
    valueblocks_t = {}
    valueblocks_t[TARGET_VAR] = values[TARGET_VAR]
    for i in range(int((n / 2) + 1)):
        if i < int(n/2):
            q = gp_.to_sparql_deep_narrow_path_query(
                hop[i], node[i+1], valueblocks_s, gp_helper[:i+1], SOURCE_VAR, gp_in=gp_in
            )
            logger.debug(q)
            try:
                t, res_q[i] = run_query(q)
            except:
                logger.debug('Die Query (s.o.) hat nicht geklappt')
                return []
            if not res_q[i]:
                logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
                return []
            elif not res_q[i]['results']['bindings']:
                logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
                return []
            values[hop[i]] = get_values([hop[i]], res_q[i])
            valueblocks_s[hop[i]] = {
                (hop[i],): random.sample(
                    values[hop[i]][(hop[i],)],
                    min(10, len(values[hop[i]][(hop[i],)]))
                )
            }
        if n-i > i:
            q = gp_.to_sparql_deep_narrow_path_query(
                hop[n-i],
                node[n-i],
                valueblocks_t,
                gp_helper[n-i:],
                TARGET_VAR,
                gp_in=gp_in
            )
            logger.debug(q)
            try:
                t, res_q[n-i] = run_query(q)
            except:
                logger.debug('Die Query (s.o.) hat nicht geklappt')
                return []
            if not res_q[n-i]:
                logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
                return []
            elif not res_q[n-i]['results']['bindings']:
                logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
                return []
            values[hop[n-i]] = get_values([hop[n-i]], res_q[n-i])
            valueblocks_t[hop[n-i]] = {
                (hop[n-i],): random.sample(
                    values[hop[n-i]][(hop[n-i],)],
                    min(10, len(values[hop[n-i]][(hop[n-i],)]))
                )
            }

    # Query fürs Ergebnis
    # gemeinsamer source/target-block, damit nur "richtige" Pfade gefunden
    # werden
    valueblocks = {}
    for key in valueblocks_s:
        if key is not SOURCE_VAR:
            valueblocks[key] = valueblocks_s[key]
    for key in valueblocks_t:
        if key is not TARGET_VAR:
            valueblocks[key] = valueblocks_t[key]
    valueblocks['st'] = values['st']
    q = gp_.to_sparql_deep_narrow_path_inst_query(
        hop, valueblocks, gp_helper, gp_in=gp_in
    )
    logger.debug(q)
    try:
        t, res_q_inst = run_query(q)
    except:
        logger.debug('Die Query (s.o.) hat nicht geklappt')
        return []
    if not res_q_inst:
        logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
        return []
    elif not res_q_inst['results']['bindings']:
        logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
        return []
    res = []
    res_rows_path = ['results', 'bindings']
    bind = sparql_json_result_bindings_to_rdflib(
        get_path(res_q_inst, res_rows_path, default=[])
    )
    for row in bind:
        gp_res = GraphPattern([
            (node[i], get_path(row, [hop[i]]), node[i + 1]) if direct[i] == 1
            else (node[i + 1], get_path(row, [hop[i]]), node[i])
            for i in range(n + 1)
        ])
        res.append(gp_res)

    return res


# fünfte Version: filtern nach Count
def mutate_deep_narrow_5(
        gp_, gtps, n, direct=None, gp_in=False
):
    node = [SOURCE_VAR]
    for i in range(n):
        node.append(Variable('n%i' % i))
    node.append(TARGET_VAR)
    hop = []
    for i in range(n + 1):
        hop.append(Variable('p%i' % i))
    if direct is None or len(direct) != n + 1:
        logger.debug(
            'No direction chosen, or direction tuple with false length'
        )
        direct = []
        for i in range(n + 1):
            direct.append(0)
    gp_helper = []
    for i in range(n + 1):
        if direct[i] == 0:
            direct[i] = random.choice([-1, 1])
        if direct[i] == 1:
            gp_helper.append(
                GraphPattern([(node[i], hop[i], node[i + 1])])
            )
        else:
            gp_helper.append(
                GraphPattern([(node[i + 1], hop[i], node[i])])
            )
    values = {}
    values[SOURCE_VAR] = {(SOURCE_VAR,): [(tup[0],) for tup in gtps]}
    values[TARGET_VAR] = {(TARGET_VAR,): [(tup[1],) for tup in gtps]}
    values['st'] = {(SOURCE_VAR, TARGET_VAR): gtps}
    res_q = []
    for i in range(n + 1):
        res_q.append({})

    # Queries für die Schritte
    valueblocks = {}
    valueblocks[SOURCE_VAR] = values[SOURCE_VAR]
    for i in range(n+1):
        q = gp_.to_sparql_deep_narrow_path_query(
            hop[i], node[i+1], valueblocks, gp_helper[:i+1], gp_in=gp_in
        )
        logger.debug(q)
        try:
            t, res_q[i] = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not res_q[i]:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not res_q[i]['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []
        values[hop[i]] = get_values([hop[i]], res_q[i])
        valueblocks[hop[i]] = get_weighted_sample(
            hop[i], Variable('avgc'+''.join(node[i+1])), res_q[i]
        )

    # Query fürs Ergebnis
    gp_help = GraphPattern([
                    (node[i], hop[i], node[i+1]) if direct[i] == 1
                    else (node[i+1], hop[i], node[i])
                    for i in range(n+1)
                    ])
    # gemeinsamer source/target-block, damit nur "richtige" Pfade gefunden
    # werden
    del valueblocks[SOURCE_VAR]
    valueblocks['st'] = values['st']
    q = gp_.to_sparql_deep_narrow_path_inst_query_old(hop, valueblocks, gp_help, gp_in=gp_in)
    logger.debug(q)
    try:
        t, res_q_inst = run_query(q)
    except:
        logger.debug('Die Query (s.o.) hat nicht geklappt')
        return []
    if not res_q_inst:
        logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
        return []
    elif not res_q_inst['results']['bindings']:
        logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
        return []
    res = []
    res_rows_path = ['results', 'bindings']
    bind = sparql_json_result_bindings_to_rdflib(
        get_path(res_q_inst, res_rows_path, default=[])
    )
    for row in bind:
        gp_res = GraphPattern([
            (node[i], get_path(row, [hop[i]]), node[i + 1]) if direct[i] == 1
            else (node[i + 1], get_path(row, [hop[i]]), node[i])
            for i in range(n + 1)
        ])
        res.append(gp_res)

    return res


# sechste Version: Query für letzten step bekommt schon die Targets
#  => Precheck feasible?
def mutate_deep_narrow_6(
        gp_, gtps, n, direct=None, gp_in=False
):
    node = [SOURCE_VAR]
    for i in range(n):
        node.append(Variable('n%i' % i))
    node.append(TARGET_VAR)
    hop = []
    for i in range(n + 1):
        hop.append(Variable('p%i' % i))
    if direct is None or len(direct) != n + 1:
        logger.debug(
            'No direction chosen, or direction tuple with false length'
        )
        direct = []
        for i in range(n + 1):
            direct.append(0)
    gp_helper = []
    for i in range(n + 1):
        if direct[i] == 0:
            direct[i] = random.choice([-1, 1])
        if direct[i] == 1:
            gp_helper.append(
                GraphPattern([(node[i], hop[i], node[i + 1])])
            )
        else:
            gp_helper.append(
                GraphPattern([(node[i + 1], hop[i], node[i])])
            )
    values = {}
    values[SOURCE_VAR] = {(SOURCE_VAR,): [(tup[0],) for tup in gtps]}
    values[TARGET_VAR] = {(TARGET_VAR,): [(tup[1],) for tup in gtps]}
    values['st'] = {(SOURCE_VAR, TARGET_VAR): gtps}
    res_q = []
    for i in range(n + 1):
        res_q.append({})

    # Pre-check:
    gp_help = GraphPattern([
                    (node[i], hop[i], node[i+1]) if direct[i] == 1
                    else (node[i+1], hop[i], node[i])
                    for i in range(n+1)
                    ])
    q = gp_help.to_sparql_precheck_query(values['st'], gp_in=gp_in)
    logger.debug(q)
    try:
        t, res_q = run_query(q)
    except:
        logger.info('Pre-Check hat nicht geklappt')
    if not res_q:
        logger.info('Pre-Check hat kein Ergebnis')
    elif not res_q['results']['bindings']:
        logger.info('Pre-Check hat keine gebundenen Variablen')
    else:
        logger.info('Pre-Check hat einen Treffer')

    # Queries für die Schritte
    valueblocks = {}
    valueblocks[SOURCE_VAR] = values[SOURCE_VAR]
    for i in range(n):
        q = gp_.to_sparql_deep_narrow_path_query(
            hop[i], node[i+1], valueblocks, gp_helper[:i+1], gp_in=gp_in
        )
        logger.debug(q)
        try:
            t, res_q[i] = run_query(q)
        except:
            logger.debug('Die Query (s.o.) hat nicht geklappt')
            return []
        if not res_q[i]:
            logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
            return []
        elif not res_q[i]['results']['bindings']:
            logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
            return []
        values[hop[i]] = get_values([hop[i]], res_q[i])
        valueblocks[hop[i]] = {
            (hop[i],): random.sample(
                values[hop[i]][(hop[i],)],
                min(10, len(values[hop[i]][(hop[i],)]))
            )
        }

    # gemeinsamer source/target-block, damit nur "richtige" Pfade gefunden
    # werden
    del valueblocks[SOURCE_VAR]
    valueblocks['st'] = values['st']
    q = gp_.to_sparql_deep_narrow_path_inst_query(
        hop, valueblocks, gp_helper, gp_in=gp_in
    )
    logger.debug(q)
    try:
        t, res_q_inst = run_query(q)
    except:
        logger.debug('Die Query (s.o.) hat nicht geklappt')
        return []
    if not res_q_inst:
        logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
        return []
    elif not res_q_inst['results']['bindings']:
        logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
        return []
    res = []
    res_rows_path = ['results', 'bindings']
    bind = sparql_json_result_bindings_to_rdflib(
        get_path(res_q_inst, res_rows_path, default=[])
    )
    for row in bind:
        gp_res = GraphPattern([
            (node[i], get_path(row, [hop[i]]), node[i + 1]) if direct[i] == 1
            else (node[i + 1], get_path(row, [hop[i]]), node[i])
            for i in range(n + 1)
        ])
        res.append(gp_res)

    return res


# siebte Version: BIDI with instantiation in last step + ws-sampling
def mutate_deep_narrow_7(
        gp_, gtps, n, direct=None, gp_in=False
):
    node = [SOURCE_VAR]
    for i in range(n):
        node.append(Variable('n%i' % i))
    node.append(TARGET_VAR)
    hop = []
    for i in range(n + 1):
        hop.append(Variable('p%i' % i))
    if direct is None or len(direct) != n + 1:
        logger.debug(
            'No direction chosen, or direction tuple with false length'
        )
        direct = []
        for i in range(n + 1):
            direct.append(0)
    gp_helper = []
    for i in range(n + 1):
        if direct[i] == 0:
            direct[i] = random.choice([-1, 1])
        if direct[i] == 1:
            gp_helper.append(
                GraphPattern([(node[i], hop[i], node[i + 1])])
            )
        else:
            gp_helper.append(
                GraphPattern([(node[i + 1], hop[i], node[i])])
            )
    values = {}
    values[SOURCE_VAR] = {(SOURCE_VAR,): [(tup[0],) for tup in gtps]}
    values[TARGET_VAR] = {(TARGET_VAR,): [(tup[1],) for tup in gtps]}
    values['st'] = {(SOURCE_VAR, TARGET_VAR): gtps}
    res_q = []
    for i in range(n+1):
        res_q.append({})

    # Queries für die Schritte
    valueblocks_s = {}
    valueblocks_s[SOURCE_VAR] = values[SOURCE_VAR]
    valueblocks_t = {}
    valueblocks_t[TARGET_VAR] = values[TARGET_VAR]
    for i in range(int((n / 2) + 1)):
        if i < int(n/2):
            q = gp_.to_sparql_deep_narrow_path_query(
                hop[i], node[i+1], valueblocks_s, gp_helper[:i+1], gp_in=gp_in
            )
            logger.debug(q)
            try:
                t, res_q[i] = run_query(q)
            except:
                logger.debug('Die Query (s.o.) hat nicht geklappt')
                return []
            if not res_q[i]:
                logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
                return []
            elif not res_q[i]['results']['bindings']:
                logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
                return []
            values[hop[i]] = get_values([hop[i]], res_q[i])
            valueblocks_s[hop[i]] = get_weighted_sample(
                hop[i], Variable('avgc' + ''.join(node[i + 1])), res_q[i]
            )
        if n-i > i:
            q = gp_.to_sparql_deep_narrow_path_query(
                hop[n-i],
                node[n-i],
                valueblocks_t,
                gp_helper[n-i:],
                startvar=TARGET_VAR,
                gp_in=gp_in
            )
            logger.debug(q)
            try:
                t, res_q[n-i] = run_query(q)
            except:
                logger.debug('Die Query (s.o.) hat nicht geklappt')
                return []
            if not res_q[n-i]:
                logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
                return []
            elif not res_q[n-i]['results']['bindings']:
                logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
                return []
            values[hop[n-i]] = get_values([hop[n-i]], res_q[n-i])
            valueblocks_t[hop[n-i]] = get_weighted_sample(
                hop[n-i], Variable('avgc' + ''.join(node[n-i])), res_q[n-i]
            )

    # Query fürs Ergebnis
    # gemeinsamer source/target-block, damit nur "richtige" Pfade gefunden
    # werden
    valueblocks = {}
    for key in valueblocks_s:
        if key is not SOURCE_VAR:
            valueblocks[key] = valueblocks_s[key]
    for key in valueblocks_t:
        if key is not TARGET_VAR:
            valueblocks[key] = valueblocks_t[key]
    valueblocks['st'] = values['st']
    q = gp_.to_sparql_deep_narrow_path_inst_query(
        hop, valueblocks, gp_helper, gp_in=gp_in
    )
    logger.debug(q)
    try:
        t, res_q_inst = run_query(q)
    except:
        logger.debug('Die Query (s.o.) hat nicht geklappt')
        return []
    if not res_q_inst:
        logger.debug('Die Query (s.o.) hat kein Ergebnis geliefert')
        return []
    elif not res_q_inst['results']['bindings']:
        logger.debug('Die Query (s.o.) hat keine gebundenen Variablen')
        return []
    res = []
    res_rows_path = ['results', 'bindings']
    bind = sparql_json_result_bindings_to_rdflib(
        get_path(res_q_inst, res_rows_path, default=[])
    )
    for row in bind:
        gp_res = GraphPattern([
            (node[i], get_path(row, [hop[i]]), node[i + 1]) if direct[i] == 1
            else (node[i + 1], get_path(row, [hop[i]]), node[i])
            for i in range(n + 1)
        ])
        res.append(gp_res)

    return res


def main():
    ground_truth_pairs = get_semantic_associations()
    ground_truth_pairs, _ = split_training_test_set(ground_truth_pairs)
    # ground_truth_pairs = ground_truth_pairs[:100]
    gtp_scores = GTPScores(ground_truth_pairs)
    res = []
    for i in range(100):
        key = random.choice(gp_found.keys())
        gp_ = gp_found[key]
        # eval_gp(gtp_scores, gp_)
        r = mutate_deep_narrow_path(sparql, timeout, gtp_scores, gp_)
        logger.info(i)
        logger.info(r)
        res.append(r)
    # for key in gp_found.keys():
    #     gp_ = gp_found[key]
    #     eval_gp(gtp_scores, gp_)
    #     for i in range(100):
    #         res_ = mutate_deep_narrow_4(
    #             gp_, gp_.matching_node_pairs, 6, gp_in=False
    #         )
    #         res.append(res_)
    #         logger.info((i, key))
    #         if res_:
    #             logger.info(res_)

    # res_eval=[]
    # res = []
    #
    # max_out = 65
    # max_in = 40
    # in_out = 'out'
    # richtung = 2
    # ground_truth_pairs = get_semantic_associations()
    # ground_truth_pairs, _ = split_training_test_set(ground_truth_pairs)
    # # ground_truth_pairs = ground_truth_pairs[0:200]
    # gtp_scores = GTPScores(ground_truth_pairs)
    # gp = gp_found['140']
    # eval_gp(gtp_scores, gp)
    #
    # for i in range(20):
    #     res.append(mutate_deep_narrow_n_hops(gp, 2, max_out=max_out, in_out=in_out))
    #
    # logger.info(res)
    #
    # durchgaenge = []
    #
    # for richtung in range(1, 9):
    #     for max_out in [5, 10, 20, 30, 40, 50, 65, 75, 85, 100, 200]:
    #         for key in gp_found.keys():
    #             durchgaenge.append((richtung, max_out, key))
    #
    # random.shuffle(durchgaenge)
    #
    # for (richtung, max_out, key) in durchgaenge:
    #     logger.info('Durchgang: richtung = %s, max_out = %s, gp.key = %s' %
    #         (richtung, max_out, key)
    #     )
    #     ground_truth_pairs = get_semantic_associations()
    #     ground_truth_pairs, _ = split_training_test_set(ground_truth_pairs)
    #     # ground_truth_pairs = random.sample(ground_truth_pairs, 100)
    #     gtp_scores = GTPScores(ground_truth_pairs)
    #     gp = gp_found[key]
    #     eval_gp(gtp_scores, gp)
    #
    #     res_gp = mutate_deep_narrow_two_hops(
    #         gp,
    #         max_out=max_out,
    #         max_in=max_in,
    #         in_out=in_out,
    #         richtung=richtung
    #     )
    #     res_gp.append(gp)
    #     res_eval = eval_gp_list(gtp_scores, res_gp)
    #     gp_eval = res_eval[-1]
    #     res_eval = sorted(
    #         res_eval[:-1], key=lambda gp_: -gp_.fitness.values.score
    #     )
    #     if res_eval:
    #         logger.info(max_out)
    #         print_graph_pattern(gp)
    #         for gp_ in res_eval:
    #             print_graph_pattern(gp_)
    #         res.append((richtung, key, max_out, gp_eval, res_eval))

    # f = open('store.pckl', 'wb')
    # pickle.dump(res, f)
    # f.close()

    # in der Konsole das res nochmal anschauen:
    # import pickle
    # f = open('tests/store.pckl', 'rb')
    # res = pickle.load(f)
    # f.close()

    # print('HERE STARTS THE RES_PRINTING:')
    # for r in res:
    #     print('richtung %s, key %s, max_out %s\n' % r[0:3])
    #     print('Original GP:\n')
    #     print_graph_pattern(r[3], print_matching_node_pairs=0)
    #     print('Top 3 found (if 3 where found, else all found) GP:\n')
    #     for i in range(min(3, len(r[4]))):
    #         print_graph_pattern(r[4][i], print_matching_node_pairs=0)

    # ground_truth_pairs = get_semantic_associations()
    # ground_truth_pairs, _ = split_training_test_set(ground_truth_pairs)
    # ground_truth_pairs = random.sample(ground_truth_pairs, 100)
    # gtp_scores = GTPScores(ground_truth_pairs)
    # gp = gp_found[random.choice(gp_found.keys())]
    #
    # max_out = 50
    # max_in = 40
    # in_out = 'out'
    #
    # res = mutate_deep_narrow_one_hop_s_t_without_direction(
    #     gp,
    #     ground_truth_pairs,
    #     max_out=max_out,
    #     max_in=max_in,
    #     in_out=in_out
    # )
    # res.append(gp)
    # res_eval = eval_gp_list(gtp_scores, res)
    # gp_eval = res_eval[-1]
    # res_eval = sorted(res_eval[:-1], key=lambda gp_: -gp_.fitness.values.score)
    #
    # print_graph_pattern(gp_eval)
    # for gp_ in res_eval:
    #     print_graph_pattern(gp_)

    # # Zählfelder für die Statistik (Zugriff über max_in_out)
    # # durchschnittliche Anzahl der zurückgegebenen pattern
    # avg_num_pat = {}
    # # maximal zurückgegebene pattern
    # max_num_pat = {}
    # # durchschnittlicher Score aller zurückgegebenen pattern
    # avg_score_all_pat = {}
    # # durchschnittlicher Score des besten zurückgegegebenen pattern
    # # (wenn vorhanden)
    # avg_score_best_pat = {}
    # # druchschnittlicher Score des besten zurückgegebenen patterns
    # # (0 wenn keins vorhanden)
    # avg_score_best_pat_pun = {}
    # # maximaler Score eines zurückgegebenen patterns
    # max_score_ovrall = {}
    # # Wie oft wurde kein pattern zurückgegeben
    # num_no_pattern = {}
    # # durchschnittliche abweichung des besten patterns vom Score des
    # # Ausgangspatterns, wenn vorhanden
    # avg_diff_all_pat = {}
    # # durchschnittliche Abweichung vom Score des Ausgangspatterns,
    # # wenn vorhanden
    # avg_diff_best_pat = {}
    # # aufaddierter score von Durchgängen ohne pattern
    # punish_avg_diff_best_pat = {}
    # # aufaddierter score von Durchgängen ohne pattern mal der durchschnittlichen
    # # Anzahl zurückgegebener pattern
    # punish_avg_diff_all_pat = {}
    # # durchschnittliche Abweichung des besten patterns vom score des
    # # Ausgangspatterns mit Strafe für gar kein pattern
    # avg_diff_all_pat_punished = {}
    # # durchschnittliche Abweichung vom Score des Ausgangspatterns, mit Strafe
    # # für gar kein pattern
    # avg_diff_best_pat_punished = {}
    # # die fünf besten (am stärksten verbessernden) pattern
    # five_best_pattern = {}
    #
    # max_out_steps = [10, 15, 20, 25, 30, 40, 50, 75, 100]
    #
    # for j in max_out_steps:
    #     avg_num_pat[j] = 0
    #     max_num_pat[j] = 0
    #     avg_score_all_pat[j] = 0
    #     avg_score_best_pat[j] = 0
    #     avg_score_best_pat_pun[j] = 0
    #     max_score_ovrall[j] = 0
    #     num_no_pattern[j] = 0
    #     avg_diff_all_pat[j] = 0
    #     avg_diff_best_pat[j] = 0
    #     punish_avg_diff_best_pat[j] = 0
    #     punish_avg_diff_all_pat[j] = 0
    #     avg_diff_all_pat_punished[j] = 0
    #     avg_diff_best_pat_punished[j] = 0
    #     five_best_pattern[j] = []
    #
    # reps = 50
    #
    # for i in range(reps):
    #     ground_truth_pairs = get_semantic_associations()
    #     ground_truth_pairs, _ = split_training_test_set(ground_truth_pairs)
    #     ground_truth_pairs = random.sample(ground_truth_pairs, 100)
    #     gtp_scores = GTPScores(ground_truth_pairs)
    #     gp = gp_found[random.choice(gp_found.keys())]
    #     for j in max_out_steps:
    #         res = mutate_deep_narrow_one_hop_s_t_without_direction(
    #             gp, ground_truth_pairs, max_out=j, in_out='out'
    #         )  # TODO: warum kommt oben None rein???
    #         res.append(gp)
    #         res_eval = eval_gp_list(gtp_scores, res)
    #         gp_eval = res_eval[-1]
    #         res_eval = sorted(
    #             res_eval[:-1], key=lambda gp_: -gp_.fitness.values.score
    #         )
    #
    #         # Statistik:
    #         avg_num_pat[j] = avg_num_pat[j] + len(res_eval) / reps
    #         if len(res_eval) > max_num_pat[j]:
    #             max_num_pat[j] = len(res_eval)
    #         for gp_ in res_eval:
    #             avg_score_all_pat[j] = avg_score_all_pat[j] + \
    #                                    gp_.fitness.values.score / \
    #                                    (len(res_eval) * reps)
    #         if res_eval:
    #             avg_score_best_pat[j] = avg_score_best_pat[j] + \
    #                                     res_eval[0].fitness.values.score
    #         if res_eval:
    #             if res_eval[0].fitness.values.score > max_score_ovrall[j]:
    #                 max_score_ovrall[j] = res_eval[0].fitness.values.score
    #         if len(res_eval) == 0:
    #             num_no_pattern[j] = num_no_pattern[j] + 1
    #         if res_eval:
    #             avg_diff_all_pat[j] = avg_diff_all_pat[j] + \
    #                                   (res_eval[0].fitness.values.score -
    #                                    gp_eval.fitness.values.score) / \
    #                                   reps
    #         for gp_ in res_eval:
    #             avg_diff_best_pat[j] = avg_diff_best_pat[j] + \
    #                                    (gp_.fitness.values.score -
    #                                     gp_eval.fitness.values.score) / \
    #                                    (len(res_eval) * reps)
    #         if not res_eval:
    #             punish_avg_diff_best_pat[j] = punish_avg_diff_best_pat[j] + \
    #                                           gp_eval.fitness.values.score
    #         if res_eval:
    #             if len(five_best_pattern[j]) < 5:
    #                 five_best_pattern[j].append((
    #                     res_eval[0].fitness.values.score -
    #                     gp_eval.fitness.values.score,
    #                     res_eval[0],
    #                     gp_eval
    #                 ))
    #                 five_best_pattern[j] = sorted(
    #                     five_best_pattern[j],
    #                     key=lambda tup_: -tup_[0]
    #                 )
    #             else:
    #                 five_best_pattern[j][4] = (
    #                     res_eval[0].fitness.values.score -
    #                     gp_eval.fitness.values.score,
    #                     res_eval[0],
    #                     gp_eval
    #                 )
    #                 five_best_pattern[j] = sorted(
    #                     five_best_pattern[j],
    #                     key=lambda tup_: -tup_[0]
    #                 )
    #         logger.info('Runde %s, min_max = %s' % (i, j))
    #         print_graph_pattern(gp)
    #         if res_eval:
    #             print_graph_pattern(res_eval[0])
    #
    # # print out the five best patterns per min_max:
    # logger.info(' The five best new patterns (per min_max): ')
    # for j in max_out_steps:
    #     for i in range(len(five_best_pattern[j])):
    #         print('min_max: %s\n' % j)
    #         print('Differenz: %s\n' % five_best_pattern[j][i][0])
    #         print_graph_pattern(five_best_pattern[j][i][1])
    #         print_graph_pattern(five_best_pattern[j][i][2])
    #
    # # more statistics
    # for j in max_out_steps:
    #     avg_score_best_pat_pun[j] = avg_score_best_pat[j] / reps
    #     if reps - num_no_pattern[j]:
    #         avg_score_best_pat[j] = avg_score_best_pat[j] / \
    #                                 (reps - num_no_pattern[j])
    #     else:
    #         avg_score_best_pat = -1
    #     punish_avg_diff_all_pat[j] = punish_avg_diff_best_pat[j] * \
    #                                  avg_num_pat[j]
    #     avg_diff_all_pat_punished[j] = avg_diff_all_pat[j] - \
    #                                    punish_avg_diff_best_pat[j]
    #     avg_diff_best_pat_punished[j] = avg_diff_best_pat[j] - \
    #                                     punish_avg_diff_all_pat[j]
    #
    # # print the statistics
    # logger.info('min_max: %s\n'
    #             'avg_num_pat: %s\n'
    #             'max_num_pat: %s\n'
    #             'avg_score_all_pat: %s\n'
    #             'avg_score_best_pat: %s\n'
    #             'avg_score_best_pat_pun: %s\n'
    #             'max_score_ovrall: %s\n'
    #             'num_no_pattern: %s\n'
    #             'avg_diff_all_pat: %s\n'
    #             'avg_diff_best_pat: %s\n'
    #             'punish_avg_diff_best_pat: %s\n'
    #             'punish_avg_diff_all_pat: %s\n'
    #             'avg_diff_all_pat_punished: %s\n'
    #             'avg_diff_best_pat_punished: %s\n' % (
    #             ' '.join([str(x) for x in max_out_steps]),
    #             ' '.join([str(avg_num_pat[x]) for x in max_out_steps]),
    #             ' '.join([str(max_num_pat[x]) for x in max_out_steps]),
    #             ' '.join([str(avg_score_all_pat[x]) for x in max_out_steps]),
    #             ' '.join([str(avg_score_best_pat[x]) for x in max_out_steps]),
    #             ' '.join(
    #                 [str(avg_score_best_pat_pun[x]) for x in max_out_steps]
    #             ),
    #             ' '.join([str(max_score_ovrall[x]) for x in max_out_steps]),
    #             ' '.join([str(num_no_pattern[x]) for x in max_out_steps]),
    #             ' '.join([str(avg_diff_all_pat[x]) for x in max_out_steps]),
    #             ' '.join([str(avg_diff_best_pat[x]) for x in max_out_steps]),
    #             ' '.join(
    #                 [str(punish_avg_diff_best_pat[x]) for x in max_out_steps]
    #             ),
    #             ' '.join(
    #                 [str(punish_avg_diff_all_pat[x]) for x in max_out_steps]
    #             ),
    #             ' '.join(
    #                 [str(avg_diff_all_pat_punished[x]) for x in max_out_steps]
    #             ),
    #             ' '.join(
    #                 [str(avg_diff_best_pat_punished[x]) for x in max_out_steps]
    #             )
    # ))
    #
    # # TODO: Fehler finden, warum die Differenz der gp-scores in
    # five_best_patterns nicht stimmt
    #
    # res = res[0:100]
    # for res_ in res:
    #     # print('max_out:' + str(res_[1]))
    #     print_graph_pattern(res_)
    #
    #     # TODO: zweite Query auch mit SOURCE TARGET binden und gp in die query
    #     # dazunehmen, dann spar ich mir auch das suchen nach Treffern ?!


if __name__ == '__main__':
    main()
