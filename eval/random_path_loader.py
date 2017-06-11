#!/usr/bin/env python2.7
# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from rdflib import Variable

from ground_truth_tools import get_semantic_associations
from logging_config import logging
from graph_pattern import SOURCE_VAR
from graph_pattern import TARGET_VAR
from graph_pattern import GraphPattern

from eval.data_generator import generate_triples
from eval.data_loader import load_triples_into_endpoint


logger = logging.getLogger(__name__)
logger.info('init')

GTPS_FILENAME = 'data/dbpedia_random_1000_uri_pairs.csv.gz'
EVAL_DATA_GRAPH = 'urn:gp_learner:eval:data'
SPARQL_ENDPOINT = 'http://localhost:8890/sparql'


def random_path(length):
    """Returns a random path with given length between source and target.

    Paths look like:
        (?source, ?ve1, ?vn1), (?vn1, ?ve2, ?vn2), ... (?vn(l-1), ?vel, ?target)

    As every edge can be flipped randomly.
    """
    assert length > 0
    edges = [Variable('ve%d' % i) for i in range(1, length + 1)]
    nodes = [Variable('vn%d' % i) for i in range(1, length)] + [TARGET_VAR]
    s = SOURCE_VAR  # start at source
    triples = []
    for e, n in zip(edges, nodes):
        triples.append((s, e, n))
        s = n
    gp = GraphPattern([
        (o, p, s) if random.random() < .5 else (s, p, o)
        for s, p, o in triples
    ])
    return gp


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description='generate random paths',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--sparql_endpoint',
        help="the SPARQL endpoint",
        action="store",
        default=SPARQL_ENDPOINT,
    )

    parser.add_argument(
        "--gtps_filename",
        help="ground truth source target file used for training and evaluation",
        action="store",
        default=GTPS_FILENAME,
    )

    parser.add_argument(
        "--eval_data_graph",
        help="graph to store eval data in",
        action="store",
        default=EVAL_DATA_GRAPH,
    )

    parser.add_argument(
        "--no_clear_graph",
        help="don't clear eval data graph before loading",
        action="store_false",
        default=True,
        dest="clear",
    )

    parser.add_argument(
        "--no_loading",
        help="no modification of endpoint, print only",
        action="store_false",
        default=True,
        dest="load",
    )

    parser.add_argument(
        "length",
        help="length of the randomized path to create and inject",
        type=int,
    )

    args = parser.parse_args()
    return args


def path_loader(
        length,
        gtps_filename=GTPS_FILENAME,
        sparql_endpoint=SPARQL_ENDPOINT,
        eval_data_graph=EVAL_DATA_GRAPH,
        load=True,
        clear=True,
        **kwds
):
    gp = random_path(length)
    logger.info(
        'Generated random graph pattern with path length %d:\n%s' % (
            length, gp))

    # get list of semantic association pairs
    semantic_associations = get_semantic_associations(
        fn=gtps_filename,
        limit=None,
    )
    gtps = semantic_associations

    triples = generate_triples(gp, gtps)
    if load:
        load_triples_into_endpoint(
            triples,
            sparql_endpoint=sparql_endpoint,
            graph=eval_data_graph,
            clear=clear
        )

    return gp


if __name__ == '__main__':
    path_loader(**vars(parse_args()))
