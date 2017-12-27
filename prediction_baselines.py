#!/usr/bin/env python2.7
# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from collections import Counter
from math import log

from SPARQLWrapper import SPARQLWrapper
from rdflib import URIRef
from rdflib import Literal
from rdflib import Variable
from splendid import get_path

import config
from gp_learner import find_in_prediction
from gp_learner import format_prediction_results
from graph_pattern import TARGET_VAR
from ground_truth_tools import get_semantic_associations
from ground_truth_tools import split_training_test_set
import gp_query
from utils import sparql_json_result_bindings_to_rdflib

logger = logging.getLogger(__name__)


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
        (get_path(row, [TARGET_VAR]), float(get_path(row, [Variable('score')])))
        for row in bindings]
    # print(target_scores)
    return target_scores


def query_template(name, triple):
    template_out = '''
    SELECT distinct(?target) ?score {
     %(source)s %(p)s ?target .
     %(triple)s
    }
    ORDER BY DESC(?score)
    '''

    template_in = '''
    SELECT distinct(?target) ?score {
     ?target %(p)s %(source)s .
     %(triple)s
    }
    ORDER BY DESC(?score)
    '''

    template_any = '''
    SELECT distinct(?target) ?score {
     { %(source)s %(p)s ?target .}
     UNION
     { ?target %(p)s %(source)s . }
     %(triple)s
    }
    ORDER BY DESC(?score)
    '''

    template_bidi = '''
    SELECT distinct(?target) ?score {
     %(source)s %(p)s ?target .
     ?target %(p)s %(source)s .
     %(triple)s
    }
    ORDER BY DESC(?score)
    '''
    linkage_names = ['out', 'in', 'any', 'bidi']
    tmplts = [template_out, template_in, template_any, template_bidi]

    res = {}
    for ln, tmplt in zip(linkage_names, tmplts):
        for pn, pred in [('', '[]'), ('wl_', 'dbo:wikiPageWikiLink')]:
            n = '%s_nb_%s%s' % (ln, pn, name)
            res[n] = tmplt % {
                'p': pred,
                'triple': triple,
                'source': '%(source)s',  # keep '%(source)s' for later
            }
    return res



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


def _q(sparql, timeout, query, selectors):
    t, q_res = gp_query._query(sparql, timeout, query)
    res_rows_path = ['results', 'bindings']
    bindings = sparql_json_result_bindings_to_rdflib(
        get_path(q_res, res_rows_path, default=[])
    )
    target_scores = [
        [get_path(row, [sel]) for sel in selectors]
        for row in bindings]
    return target_scores


def predict_target_with_milne_witten(
        sparql, pred, source, timeout=TIMEOUT, limit=LIMIT):
    """Predicts target with Milne-Witten Relatedness (adapted to DBpedia)."""
    N = 31394968  # SELECT count(distinct ?s) { ?s ?p ?o }
    lN = log(N)

    st_inlink_count = [(u, int(i)) for u, i in _q(
        sparql, timeout,
        'SELECT ?target (COUNT(DISTINCT ?i) as ?c) {\n'
        '  ?i %(p1)s %(source)s .\n'
        '  ?i %(p2)s ?target .'
        '  FILTER(?target != %(source)s)\n'
        '}\n'
        'GROUP BY ?target\n'
        'HAVING(COUNT(DISTINCT ?i) > 1)\n'
        'ORDER BY DESC(?c)\n'
        'LIMIT %(limit)d' % {
            'p1': pred if pred else '?p1',
            'p2': pred if pred else '?p2',
            'source': source.n3(),
            'limit': limit,
        },
        [TARGET_VAR, Variable('c')]
    )]
    # for u, i in st_inlink_count:
    #     print(i, u)
    logger.info('%s target candidates...', len(st_inlink_count))

    indeg_q = 'SELECT (COUNT(DISTINCT ?i) AS ?c) { ?i %(pred)s %(node)s . }'
    indegs = []
    for node in [source] + [u for u, _ in st_inlink_count]:
        indegs.append(int(_q(
            sparql, timeout,
            indeg_q % {
                'pred': pred if pred else '?p',
                'node': node.n3()
            },
            [Variable('c')]
        )[0][0]))
    # print(indegs)

    s_indeg = indegs.pop(0)
    scores = Counter()
    for (t, st_indeg_overlap), t_indeg in zip(st_inlink_count, indegs):
        mw = 1 - (
            (log(max(s_indeg, t_indeg)) - log(st_indeg_overlap))
            / (lN - log(min(s_indeg, t_indeg)))
        )
        scores[t] = mw
    # res = scores.most_common()
    # for t, mw in res[:10]:
    #     print('\t', mw, t)
    return scores.most_common()


def main():
    semantic_associations = get_semantic_associations(
        config.GT_ASSOCIATIONS_FILENAME)
    assocs_train, assocs_test = split_training_test_set(
        semantic_associations, variant='random'
    )

    # setup node expander
    sparql = SPARQLWrapper(config.SPARQL_ENDPOINT)

    predict_list = assocs_test

    # degree, pagerank and hits
    for method, query in sorted(prediction_queries.items()):
        target_idxs = []
        for source, target in predict_list:
            logger.info(
                'method: %s, predicting targets for %s, ground truth: %s',
                method, source.n3(), target.n3())
            prediction = predict_target_with_query(sparql, query, source)
            idx = find_in_prediction(prediction, target)
            logger.info(
                format_prediction_results(method, prediction, target, idx))
            target_idxs.append(idx)
        print("'%s': %s," % (method, target_idxs))

    # milne-witten relatedness
    for method, pred in (('mw_wl', 'dbo:wikiPageWikiLink'),):
        target_idxs = []
        for source, target in predict_list:
            logger.info(
                'method: %s, predicting targets for %s, ground truth: %s',
                method, source.n3(), target.n3())
            prediction = predict_target_with_milne_witten(sparql, pred, source)
            idx = find_in_prediction(prediction, target)
            logger.info(
                format_prediction_results(method, prediction, target, idx))
            target_idxs.append(idx)
        print("'%s': %s," % (method, target_idxs))


if __name__ == '__main__':
    main()
