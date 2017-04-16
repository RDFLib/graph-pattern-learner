#!/usr/bin/env python2.7
# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random
from uuid import uuid4

from rdflib import Namespace
from rdflib import URIRef
from rdflib import Variable

from ground_truth_tools import get_semantic_associations
from ground_truth_tools import split_training_test_set
from logging_config import logging
from graph_pattern import SOURCE_VAR
from graph_pattern import TARGET_VAR
from graph_pattern import GraphPattern
from utils import exception_stack_catcher
from utils import log_all_exceptions

logger = logging.getLogger(__name__)
logger.info('init')

EVAL_DATA_GRAPH = 'urn:gp_learner:eval:data'
EVAL_DATA_NS = Namespace(EVAL_DATA_GRAPH + ':')


def generate_triples(
        gp,
        stps,
        vars_joint='none',
):
    """Generate a triple instantiation for gp wrt. the given stps.
    
    This method is intended to generate an instantiation for a gp to inject it
    into an endpoint for evaluation of the gp_learner.
    
    Given a pattern like:
    >>> from rdflib import Variable, Namespace
    >>> gp = GraphPattern((
    ...     (SOURCE_VAR, Variable('v1'), Variable('v2')),
    ...     (TARGET_VAR, Variable('v3'), Variable('v2')),
    ... ))
    
    And a list of source-target-pairs like:
    >>> dbr = Namespace('http://dbpedia.org/resource/')
    >>> stps = [(dbr['Dog'], dbr['Cat']), (dbr['Horse'], dbr['Saddle'])]
    
    This method will generate instantiations (URIRefs) for the Variables v1, v2
    and v3. Depending on the vars_joint parameter the triples for different stps
    will share URIRefs for 'none' (default), 'edges' or 'all' variables.
    
    >>> g = generate_triples(gp, stps, 'all')
    >>> l = [
    ...     (dbr['Cat'], EVAL_DATA_NS['v3'], EVAL_DATA_NS['v2']),
    ...     (dbr['Dog'], EVAL_DATA_NS['v1'], EVAL_DATA_NS['v2']),
    ...     (dbr['Horse'], EVAL_DATA_NS['v1'], EVAL_DATA_NS['v2']),
    ...     (dbr['Saddle'], EVAL_DATA_NS['v3'], EVAL_DATA_NS['v2']),
    ... ]
    >>> sorted(g) == sorted(l)
    True
    
    >>> g = generate_triples(gp, stps, 'edges')
    >>> l = sorted(g)
    >>> l[1][1] == l[2][1]  # edge URI from Dog and from Horse is same 
    True
    >>> l[1][2] != l[2][2]  # connecting node is disjoint
    True
    
    >>> g = generate_triples(gp, stps, 'none')
    >>> l = sorted(g)
    >>> l[1][1] != l[2][1]  # edge URI from Dog and from Horse is disjoint 
    True
    >>> l[1][2] != l[2][2]  # connecting node is disjoint
    True
    
    :param gp: The graph pattern to instantiate.
    :param stps: The source-target-pairs for which to fill 
    :param vars_joint: One of ('none', 'edges', 'all'), default: 'none'.
    :return: A triple generator.
    """
    assert isinstance(gp, GraphPattern)
    assert vars_joint in ('none', 'edges', 'all')
    vars_ = gp.vars_in_graph
    assert {SOURCE_VAR, TARGET_VAR} <= vars_

    if vars_joint == 'none':
        vars_ = set()
    elif vars_joint == 'edges':
        vars_ = gp.edge_vars()
    else:
        # 'all' joint, use vars_ as is from above
        pass
    vars_ = vars_ - {SOURCE_VAR, TARGET_VAR}

    # for given vars generate static URIRefs
    gp_static = gp.replace({v: EVAL_DATA_NS[v] for v in vars_})
    # for each of the remaining vars create a UUID URIRef per stp:
    dyn_vars = gp_static.vars_in_graph - {SOURCE_VAR, TARGET_VAR}

    # for each stp generate pattern instantiation
    n = 0
    for source, target in stps:
        mapping = {v: EVAL_DATA_NS[str(uuid4())] for v in dyn_vars}
        mapping.update({SOURCE_VAR: source, TARGET_VAR: target})
        for t in gp_static.replace(mapping):
            yield t
            n += 1
    logger.debug("Generated %d triples", n)


def main():
    from rdflib import Variable
    gp = GraphPattern((
        (SOURCE_VAR, Variable('v1'), Variable('v2')),
        (TARGET_VAR, Variable('v3'), Variable('v2')),
    ))
    # get list of semantic association pairs and split in train and test sets
    semantic_associations = get_semantic_associations(
        fn='data/dbpedia_random_1000k_uri_pairs.csv.gz',
        limit=100,
    )
    # assocs_train, assocs_test = split_training_test_set(
    #     semantic_associations
    # )
    # stps = tuple(sorted(assocs_train))
    stps = semantic_associations
    print(len(stps))

    triples = generate_triples(gp, stps)
    for t in triples:
        print(t)


if __name__ == '__main__':
    main()
    import doctest
    doctest.testmod()
