# encoding: utf-8
"""Online tests for gp_query.

The following tests are depending on the endpoint and its loaded datasets.
In case of errors also check the used endpoint and if the tests make sense.
"""

from rdflib import URIRef
from rdflib import Variable
import SPARQLWrapper

from config import MUTPB_FV_QUERY_LIMIT
from config import SPARQL_ENDPOINT
from gp_query import calibrate_query_timeout
from gp_query import predict_query
from gp_query import variable_substitution_query
from graph_pattern import GraphPattern
from graph_pattern import SOURCE_VAR
from graph_pattern import TARGET_VAR

import logging
logger = logging.getLogger(__name__)

sparql = SPARQLWrapper.SPARQLWrapper(SPARQL_ENDPOINT)
try:
    timeout = max(5, calibrate_query_timeout(sparql))  # 5s for warmup
except IOError:
    from nose import SkipTest
    raise SkipTest(
        "Can't establish connection to SPARQL_ENDPOINT %s, skipping tests in %s"
        % (SPARQL_ENDPOINT, __file__))

a = URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type')
wpl = URIRef('http://dbpedia.org/ontology/wikiPageWikiLink')


def test_variable_substitution_query():
    source_target_pairs = [
        (URIRef('http://dbpedia.org/resource/Adolescence'),
         URIRef('http://dbpedia.org/resource/Youth')),
        (URIRef('http://dbpedia.org/resource/Adult'),
         URIRef('http://dbpedia.org/resource/Child')),
        (URIRef('http://dbpedia.org/resource/Affinity_(law)'),
         URIRef('http://dbpedia.org/resource/Mother')),
        (URIRef('http://dbpedia.org/resource/Alchemy'),
         URIRef('http://dbpedia.org/resource/Gold')),
        (URIRef('http://dbpedia.org/resource/Alderman'),
         URIRef('http://dbpedia.org/resource/Mayor')),
        (URIRef('http://dbpedia.org/resource/Algebra'),
         URIRef('http://dbpedia.org/resource/Mathematics')),
        (URIRef('http://dbpedia.org/resource/Amen'),
         URIRef('http://dbpedia.org/resource/Prayer')),
        (URIRef('http://dbpedia.org/resource/Amnesia'),
         URIRef('http://dbpedia.org/resource/Memory')),
        (URIRef('http://dbpedia.org/resource/Angel'),
         URIRef('http://dbpedia.org/resource/Heaven')),
        (URIRef('http://dbpedia.org/resource/Arithmetic'),
         URIRef('http://dbpedia.org/resource/Mathematics')),
    ]

    gp = GraphPattern([
        (SOURCE_VAR, Variable('edge'), TARGET_VAR),
        (SOURCE_VAR, a, Variable('source_type')),
        (TARGET_VAR, a, Variable('target_type')),
    ])
    limit = MUTPB_FV_QUERY_LIMIT
    t, res = variable_substitution_query(
        sparql, timeout, gp, Variable('edge'), source_target_pairs, limit)
    logger.debug(res.most_common())
    assert res and res.most_common()[0][0] == wpl

    gp = GraphPattern([
        (Variable('var'), wpl, SOURCE_VAR),
    ])
    t, res = variable_substitution_query(
        sparql, timeout, gp, Variable('var'), source_target_pairs, limit)
    logger.debug(res.most_common())
    assert (URIRef('http://dbpedia.org/resource/Human'), 3) in res.most_common()


def test_predict_query():
    source = URIRef('http://dbpedia.org/resource/Algebra')
    target = URIRef('http://dbpedia.org/resource/Mathematics')
    gp = GraphPattern([
        (SOURCE_VAR, wpl, TARGET_VAR),
        (TARGET_VAR, wpl, SOURCE_VAR),
        (TARGET_VAR, a, Variable('target_type'))
    ])
    t, res = predict_query(sparql, timeout, gp, source)
    assert len(res) > 0
    assert target in res
