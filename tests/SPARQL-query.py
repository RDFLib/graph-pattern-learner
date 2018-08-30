# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""ein File einfach um SPARQL-queries abzufeuern, statt es online im Browser
zu machen.
"""

import logging
from collections import OrderedDict
from os import getenv

import SPARQLWrapper
from splendid import time_func
import socket
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


sparql = SPARQLWrapper.SPARQLWrapper(
    getenv('SPARQL_ENDPOINT', 'http://dbpedia.org/sparql'))
try:
    timeout = max(5, calibrate_query_timeout(sparql))  # 5s for warmup
except IOError:
    from nose import SkipTest
    raise SkipTest(
        "Can't establish connection to SPARQL_ENDPOINT:\n    %s\n"
        "Skipping tests in\n    %s" % (SPARQL_ENDPOINT, __file__))

sparql.resetQuery()
sparql.setTimeout(timeout)
sparql.setReturnFormat(SPARQLWrapper.JSON)

q = 'SELECT ?source ?target ?vcb0 ?vcb1 ?vcb2 ?vcb3 WHERE {' \
    '?source ?vcb0 ?vcb2 .' \
    '?target <http://dbpedia.org/ontology/thumbnail> ?vcb3 .' \
    '?target <http://dbpedia.org/property/image> ?vcb1 .' \
    '?vcb2 <http://dbpedia.org/ontology/wikiPageWikiLink> ?source .' \
    '?vcb2 <http://purl.org/linguistics/gold/hypernym> ?target ' \
    '}'

try:
    q_short = ' '.join((line.strip() for line in q.split('\n')))
    sparql.setQuery(q_short)
    c = time_func(sparql.queryAndConvert)
except socket.timeout:
    c = (timeout, {})
except ValueError:
    # e.g. if the endpoint gives us bad JSON for some unicode chars
    print(
        'Could not parse result for query, assuming empty result...\n'
        'Query:\n%s\nException:', q,
        exc_info=1,  # appends exception to message
    )
    c = (timeout, {})

t, res = c
print('orig query took %.4f s, result:\n%s\n', t, res)