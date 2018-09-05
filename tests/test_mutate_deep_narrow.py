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


if __name__ == '__main__':
    main()
