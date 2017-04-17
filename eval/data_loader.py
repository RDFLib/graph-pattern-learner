#!/usr/bin/env python2.7
# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import urllib2

import SPARQLWrapper
from rdflib import Graph
from rdflib import Namespace
from rdflib import URIRef
from rdflib import Variable
from splendid import chunker

from ground_truth_tools import get_semantic_associations
from ground_truth_tools import split_training_test_set
from logging_config import logging
from graph_pattern import SOURCE_VAR
from graph_pattern import TARGET_VAR
from graph_pattern import GraphPattern
from utils import exception_stack_catcher
from utils import log_all_exceptions

from eval.data_generator import generate_triples


logger = logging.getLogger(__name__)
logger.info('init')

EVAL_DATA_GRAPH = 'urn:gp_learner:eval:data'
# SPARQL_ENDPOINT = 'http://elwe4.rhrk.uni-kl.de:8890/sparql'
# SPARQL_ENDPOINT = 'http://serv-4101.kl.dfki.de:8890/sparql'
SPARQL_ENDPOINT = 'http://localhost:8890/sparql'


def load_triples_into_endpoint(
        triples,
        sparql_endpoint=SPARQL_ENDPOINT,
        graph=EVAL_DATA_GRAPH,
        batch_size=1000,
        clear=True,
):
    if clear:
        clear_graph(sparql_endpoint, graph)
    sparql = SPARQLWrapper.SPARQLWrapper(sparql_endpoint)
    sparql.setMethod(SPARQLWrapper.POST)
    logger.info(
        'Loading triples into graph %s on endpoint %s',
        graph, sparql_endpoint
    )
    n = 0
    for chunk in chunker(triples, batch_size):
        n += len(chunk)
        trips = ' .\n'.join(' '.join(i.n3() for i in t) for t in chunk) + ' .\n'
        q = 'INSERT DATA { GRAPH %s {\n%s}}' % (
            URIRef(graph).n3(),
            trips
        )
        logger.debug(q)
        sparql.setQuery(q)
        sparql.query()
    logger.info(
        'Successfully loaded %d triples into graph %s on endpoint %s',
        n, graph, sparql_endpoint
    )


def clear_graph(sparql_endpoint=SPARQL_ENDPOINT, graph=EVAL_DATA_GRAPH):
    sparql = SPARQLWrapper.SPARQLWrapper(sparql_endpoint)
    sparql.setMethod(SPARQLWrapper.POST)
    q = 'CLEAR GRAPH %s' % (URIRef(graph).n3(),)
    logger.info('Clearing graph %s on endpoint %s', graph, sparql_endpoint)
    sparql.setQuery(q)
    try:
        sparql.query()
    except urllib2.HTTPError:
        # argh, don't ask me why, but it seems we get a 406 on success
        # TODO: report to SPARQLWrapper?
        pass


def main():
    from rdflib import Variable
    gp = GraphPattern((
        (SOURCE_VAR, Variable('v1'), Variable('v2')),
        (TARGET_VAR, Variable('v3'), Variable('v2')),
    ))
    # get list of semantic association pairs and split in train and test sets
    semantic_associations = get_semantic_associations(
        fn='data/dbpedia_random_1000k_uri_pairs.csv.gz',
        limit=None,
    )
    # assocs_train, assocs_test = split_training_test_set(
    #     semantic_associations
    # )
    # stps = tuple(sorted(assocs_train))
    stps = semantic_associations

    triples = generate_triples(gp, stps)
    load_triples_into_endpoint(triples)


if __name__ == '__main__':
    main()
    import doctest
    doctest.testmod()
