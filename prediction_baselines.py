#!/usr/bin/env python2.7
# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from SPARQLWrapper import SPARQLWrapper
from rdflib import URIRef
from rdflib import Literal
from rdflib import Variable
from splendid import get_path

import config
from gp_learner import find_in_prediction
from graph_pattern import TARGET_VAR
from ground_truth_tools import get_semantic_associations
from ground_truth_tools import split_training_test_set
import gp_query
from utils import sparql_json_result_bindings_to_rdflib

TIMEOUT = 30
LIMIT = 300


def predict_target_with_query(
        sparql, query, source, timeout=TIMEOUT, limit=LIMIT):
    """Predicts target with given query.

    For example for pagerank_bidi:
    SELECT distinct(?target) ?score {
     { dbr:Circle ?p ?target .}
     UNION
     { ?target ?q dbr:Circle . }
     ?target dbo:wikiPageRank ?score
    }
    ORDER BY DESC(?score)
    LIMIT 100
    """
    q = query % {'source': source.n3()}
    q += '\nLIMIT %d' % limit
    t, q_res = gp_query._query(sparql, timeout, q)
    res_rows_path = ['results', 'bindings']
    bindings = sparql_json_result_bindings_to_rdflib(
        get_path(q_res, res_rows_path, default=[])
    )
    target_scores = [
        (get_path(row, [TARGET_VAR]), get_path(row, [Variable('score')]))
        for row in bindings]
    # print(target_scores)
    return target_scores


def query_template(name, triple):
    template_out = '''
    SELECT distinct(?target) ?score {
     %(source)s ?p ?target .
     %(triple)s
    }
    ORDER BY DESC(?score)
    '''

    template_in = '''
    SELECT distinct(?target) ?score {
     ?target ?p %(source)s .
     %(triple)s
    }
    ORDER BY DESC(?score)
    '''

    template_bidi = '''
    SELECT distinct(?target) ?score {
     { %(source)s ?p ?target .}
     UNION
     { ?target ?q %(source)s . }
     %(triple)s
    }
    ORDER BY DESC(?score)
    '''

    names = map(lambda s: name + '_%s' % s, ['out', 'in', 'bidi'])
    replace = {
        'triple': triple,
        'source': '%(source)s',  # we want to keep '%(source)s' for later
    }
    queries = map(lambda t: t % replace,
                  [template_out, template_in, template_bidi])
    return dict(zip(names, queries))



prediction_queries = {}
prediction_queries.update(
    query_template('pagerank', '?target dbo:wikiPageRank ?score .')
)
prediction_queries.update(
    query_template('hits', '?target dbo:wikiHITS ?score .')
)
prediction_queries.update(
    query_template('outdeg', '?target dbo:wikiPageOutLinkCountCleaned ?score .')
)
prediction_queries.update(
    query_template('indeg', '?target dbo:wikiPageInLinkCountCleaned ?score .')
)



def main():
    semantic_associations = get_semantic_associations(
        config.GT_ASSOCIATIONS_FILENAME)
    assocs_train, assocs_test = split_training_test_set(
        semantic_associations, variant='random'
    )

    # setup node expander
    sparql = SPARQLWrapper(config.SPARQL_ENDPOINT)

    predict_set = assocs_test



    for method, query in sorted(prediction_queries.items()):
        target_idxs = []
        for source, target in predict_set:
            prediction = predict_target_with_query(sparql, query, source)
            target_idxs.append(find_in_prediction(prediction, target))
        print("'%s': %s," % (method, target_idxs))

if __name__ == '__main__':
    main()
