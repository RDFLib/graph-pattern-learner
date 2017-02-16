#!/usr/bin/env python2.7
# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
from collections import Counter
from collections import Sequence
import logging
import re
import socket
from time import sleep

from cachetools import LRUCache
import six
from rdflib.term import Identifier
import SPARQLWrapper
from SPARQLWrapper.SPARQLExceptions import EndPointNotFound
from SPARQLWrapper.SPARQLExceptions import SPARQLWrapperException
from xml.sax.expatreader import SAXParseException
# noinspection PyUnresolvedReferences
from six.moves.urllib.error import URLError
from splendid import chunker
from splendid import get_path
from splendid import time_func

import config
from graph_pattern import GraphPattern
from graph_pattern import SOURCE_VAR
from graph_pattern import TARGET_VAR
from graph_pattern import ASK_VAR
from graph_pattern import COUNT_VAR
from graph_pattern import NODE_VAR_SUM
from graph_pattern import EDGE_VAR_COUNT
from utils import exception_stack_catcher
from utils import sparql_json_result_bindings_to_rdflib
from utils import timer

logger = logging.getLogger(__name__)




class EvalException(Exception):
    pass


class QueryException(EvalException):
    pass


def calibrate_query_timeout(
        sparql, factor=config.QUERY_TIMEOUT_FACTOR, q=None, n_queries=10):
    assert isinstance(sparql, SPARQLWrapper.SPARQLWrapper)
    logger.debug('calibrating query timer')
    total_time = 0.
    if q is None:
        q = 'select * where { ?s ?p ?o } limit 10'
    for _ in range(n_queries):
        sparql.resetQuery()
        sparql.setReturnFormat(SPARQLWrapper.JSON)
        sparql.setQuery(q)
        t, r = time_func(sparql.queryAndConvert)
        total_time += t
    avg = total_time / n_queries
    timeout = avg * factor
    timeout = max(timeout, config.QUERY_TIMEOUT_MIN)
    logger.info('Query timeout calibration: %d simplistic queries took %.3fs '
                '(%.3fs avg). Setting timeout to %.3fs',
                n_queries, total_time, avg, timeout)
    return timeout


def query_time_soft_exceeded(t, timeout):
    return t > timeout * 3 / 4


def query_time_hard_exceeded(t, timeout):
    return t >= timeout


def _get_vars_values_mapping(graph_pattern, source_target_pairs):
    vars_in_graph = graph_pattern.vars_in_graph
    if SOURCE_VAR in vars_in_graph and TARGET_VAR in vars_in_graph:
        _vars = (SOURCE_VAR, TARGET_VAR)
        _values = source_target_pairs
        _ret_val_mapping = {stp: [stp] for stp in source_target_pairs}
    else:
        sources, targets = zip(*source_target_pairs)
        _ret_val_mapping = defaultdict(list)
        if SOURCE_VAR in vars_in_graph:
            _vars = (SOURCE_VAR,)
            _values = [(s,) for s in sorted(set(sources))]
            _val_idx = 0
        elif TARGET_VAR in vars_in_graph:
            _vars = (TARGET_VAR,)
            _values = [(t,) for t in sorted(set(targets))]
            _val_idx = 1
        else:
            raise QueryException(
                "tried to run a query on a graph pattern without "
                "%s and %s vars:\n%s" % (SOURCE_VAR, TARGET_VAR, graph_pattern)
            )
        # remember which source_target_pair is reached via which value
        for stp in source_target_pairs:
            _ret_val_mapping[(stp[_val_idx],)].append(stp)
    return _vars, _values, _ret_val_mapping


def ask_multi_query(
        sparql, timeout, graph_pattern, source_target_pairs,
        batch_size=config.BATCH_SIZE):
    assert isinstance(source_target_pairs, Sequence)
    assert isinstance(source_target_pairs[0], tuple)
    _vars, _values, _ret_val_mapping = _get_vars_values_mapping(
        graph_pattern, source_target_pairs)
    # see stats_for_paper
    # assert _vars != (SOURCE_VAR, TARGET_VAR), \
    #     "why run an ask_multi_query on a complete graph pattern when you " \
    #     "could run a combined_ask_count_multi_query?"

    return _multi_query(
        sparql, timeout, graph_pattern, source_target_pairs, batch_size,
        _vars, _values, _ret_val_mapping,
        _ask_res_init, _ask_chunk_query_creator, _ask_chunk_result_extractor
    )


def _ask_res_init(source_target_pairs):
    return {stp: False for stp in source_target_pairs}


def _ask_chunk_query_creator(gp, _vars, values_chunk):
    return gp.to_sparql_select_query(
        projection=_vars,
        distinct=True,
        values={_vars: values_chunk},
    )


def _ask_chunk_result_extractor(q_res, _vars, _ret_val_mapping):
    chunk_res = {}
    res_rows_path = ['results', 'bindings']
    bindings = sparql_json_result_bindings_to_rdflib(
        get_path(q_res, res_rows_path, default=[])
    )
    for row in bindings:
        row_res = tuple([get_path(row, [v]) for v in _vars])
        stps = _ret_val_mapping[row_res]
        chunk_res.update({stp: True for stp in stps})
    return chunk_res


# noinspection PyBroadException
def _multi_query(
        sparql, timeout, graph_pattern, source_target_pairs,
        batch_size,
        _vars, _values, _ret_val_mapping,
        _res_init, _chunk_q, _chunk_res,
        _res_update=lambda r, u, **___: r.update(u),
        **kwds):
    total_time = 0
    res = _res_init(source_target_pairs, **kwds)
    for val_chunk in chunker(_values, batch_size):
        q = _chunk_q(graph_pattern, _vars, val_chunk, **kwds)
        chunk_stps = [stp for v in val_chunk for stp in _ret_val_mapping[v]]
        _start_time = timer()
        t = None
        chunk_res = None
        loop = 1
        while loop:
            loop -= 1
            try:
                t, q_res = _query(sparql, timeout, q, **kwds)
                chunk_res = _chunk_res(
                    q_res, _vars, _ret_val_mapping, **kwds)
            except EndPointNotFound:
                # happens if the endpoint reports a 404...
                # as virtuoso in rare cases seems to report a 404 let's
                # retry once after some time but then
                if not loop:  # expected to 0 on first such exception
                    logger.info(
                        'SPARQL endpoint reports a 404, will retry once in 10s'
                    )
                    sleep(10)
                    loop += 2
                    continue
                else:  # expected to be 1 on second such exception
                    loop = 0
                    logger.warning(
                        'SPARQL endpoint unreachable even after back-off '
                        'and retry\n'
                        'could not perform query:\n%s for %s\nException:',
                        q, val_chunk,
                        exc_info=1,  # appends exception to message
                    )
                    t, chunk_res = timer() - _start_time, {}
            except (SPARQLWrapperException, SAXParseException, URLError) as e:
                if (isinstance(e, SPARQLWrapperException) and
                        re.search(
                            r'The estimated execution time [0-9]+ \(sec\) '
                            r'exceeds the limit of [0-9]+ \(sec\)\.',
                            repr(e))):
                    t, chunk_res = timeout, {}
                elif len(val_chunk) > 1:
                    logger.debug('error in batch: {}'.format(val_chunk))
                    logger.debug('retrying with half size batch: {}...'.format(
                        len(val_chunk) // 2
                    ))
                    t, chunk_res = _multi_query(
                        sparql, timeout, graph_pattern, chunk_stps,
                        len(val_chunk) // 2,
                        _vars, val_chunk, _ret_val_mapping,
                        _res_init, _chunk_q, _chunk_res,
                        _res_update,
                        **kwds)
                else:
                    logger.warning(
                        'could not perform query:\n%s for %s\nException:',
                        q, val_chunk,
                        exc_info=1,  # appends exception to message
                    )
                    t, chunk_res = timer() - _start_time, {}
            except Exception:
                # TODO: maybe introduce a max error counter? per process?
                logger.warning(
                    'unhandled exception, assuming empty res for multi-query:\n'
                    'Query:\n%s\nChunk:%r\nException:',
                    q, val_chunk,
                    exc_info=1,  # appends exception to message
                )
                t, chunk_res = timer() - _start_time, {}
        _res_update(res, chunk_res, **kwds)
        total_time += t
        if query_time_soft_exceeded(total_time, timeout):
            logger.debug('early terminating batch query as timeout/2 exceeded')
            break
    return total_time, res


def combined_ask_count_multi_query(
        sparql, timeout, graph_pattern, source_target_pairs,
        batch_size=config.BATCH_SIZE // 2):
    _vars, _values, _ret_val_mapping = _get_vars_values_mapping(
        graph_pattern, source_target_pairs)
    assert _vars == (SOURCE_VAR, TARGET_VAR), \
        "combined_ask_count_multi_query on incomplete pattern?"

    t, res = _multi_query(
        sparql, timeout, graph_pattern, source_target_pairs, batch_size,
        _vars, _values, _ret_val_mapping,
        _combined_res_init, _combined_chunk_q, _combined_chunk_res
    )
    ask_res = {stp: a for stp, (a, _) in res.items()}
    count_res = {stp: c for stp, (_, c) in res.items()}
    return t, (ask_res, count_res)


def _combined_res_init(source_target_pairs):
    return {stp: (0, 0) for stp in source_target_pairs}


def _combined_chunk_q(gp, _vars, values_chunk):
    return gp.to_combined_ask_count_query(values={_vars: values_chunk})


def _combined_chunk_res(q_res, _vars, _ret_val_mapping):
    chunk_res = {}
    res_rows_path = ['results', 'bindings']
    bindings = sparql_json_result_bindings_to_rdflib(
        get_path(q_res, res_rows_path, default=[])
    )
    for row in bindings:
        row_res = tuple([get_path(row, [v]) for v in _vars])
        stps = _ret_val_mapping[row_res]
        ask_res = int(get_path(row, [ASK_VAR], '0'))
        count_res = int(get_path(row, [COUNT_VAR], '0'))
        chunk_res.update({stp: (ask_res, count_res) for stp in stps})
    return chunk_res


def count_query(sparql, timeout, graph_pattern, source=None,
                **kwds):
    assert isinstance(graph_pattern, GraphPattern)
    assert source is None or isinstance(source, Identifier)

    bind = {}
    projection = []
    count = (COUNT_VAR, TARGET_VAR)
    vars_in_graph = graph_pattern.vars_in_graph
    if SOURCE_VAR in vars_in_graph and source:
        bind[SOURCE_VAR] = source
    assert TARGET_VAR in vars_in_graph, \
        'count query without ?target in graph pattern? what to count?'

    q = graph_pattern.to_sparql_select_query(
        projection=projection,
        distinct=True,
        count=count,
        bind=bind,
    )
    try:
        res = _query(sparql, timeout, q, **kwds)
    except (SPARQLWrapperException, SAXParseException, URLError):
        res = timeout, {}
    return res


@exception_stack_catcher
def predict_query(sparql, timeout, graph_pattern, source,
                  limit=config.PREDICTION_RESULT_LIMIT):
    """Performs a single query starting at ?SOURCE returning all ?TARGETs."""
    assert isinstance(graph_pattern, GraphPattern)
    assert isinstance(source, Identifier)

    vars_in_graph = graph_pattern.vars_in_graph
    if TARGET_VAR not in vars_in_graph:
        logger.warning(
            'graph pattern without %s used for prediction:\n%r',
            TARGET_VAR.n3(), graph_pattern
        )
        return timeout, []

    q = graph_pattern.to_sparql_select_query(
        projection=[TARGET_VAR],
        distinct=True,
        bind={SOURCE_VAR: source},
        limit=limit,
    )
    try:
        t, q_res = _query(sparql, timeout, q)
    except (SPARQLWrapperException, SAXParseException, URLError):
        logger.warning(
            'Exception occurred during prediction, assuming empty result...\n'
            'Query:\n%s\nException:', q,
            exc_info=1,  # appends exception to message
        )
        t, q_res = timeout, {}
    else:
        if query_time_soft_exceeded(t, timeout):
            kind = 'hard' if query_time_hard_exceeded(t, timeout) else 'soft'
            logger.info(
                'prediction query exceeded %s timeout %s:\n%s',
                kind, t, q
            )

    res = []
    res_rows_path = ['results', 'bindings']
    bindings = sparql_json_result_bindings_to_rdflib(
        get_path(q_res, res_rows_path, default=[])
    )
    for row in bindings:
        res.append(get_path(row, [TARGET_VAR]))
    return timeout, set(res)


def _query(
        sparql, timeout, q, cache=LRUCache(maxsize=config.CACHE_SIZE), **_):
    """Cached low level function to perform a single SPARQL query.

    :param sparql: SPARQLWrapper endpoint
    :param timeout: a timeout in seconds. The endpoint 'timeout' parameter will
        be set to 3/4 this value in ms (Virtuoso seems to treat non zero
        timeouts < 1000ms as 1000ms), instructing the server to give us a
        partial result up to this soft limit. We also set a hard timeout via the
        socket to really cancel the request if there's no result after timeout
        seconds.
    :param q: The SPARQL Query as string
    :param cache: a cache object like cachetools.LRUCache or None
    :return: a (t, res) pair with querytime t as float and res as dict.
    """

    assert isinstance(sparql, SPARQLWrapper.SPARQLWrapper)
    assert isinstance(q, six.string_types)
    sparql.resetQuery()
    sparql.setTimeout(timeout)
    sparql.setReturnFormat(SPARQLWrapper.JSON)
    # sparql.setMethod(SPARQLWrapper.POST)
    # sparql.setRequestMethod(SPARQLWrapper.POSTDIRECTLY)

    # set query timeout parameter to half the hard timeout time
    sparql.addParameter('timeout', str(int(timeout * 1000 * 3 / 4)))

    logger.debug('performing sparql query: \n%s', q)
    c = cache.get(q) if cache is not None else None
    if c is None:
        logger.debug('cache miss')
        try:
            q_short = ' '.join((line.strip() for line in q.split('\n')))
            sparql.setQuery(q_short)
            c = time_func(sparql.queryAndConvert)
        except socket.timeout:
            c = (timeout, {})
        except ValueError:
            # e.g. if the endpoint gives us bad JSON for some unicode chars
            logger.warning(
                'Could not parse result for query, assuming empty result...\n'
                'Query:\n%s\nException:', q,
                exc_info=1,  # appends exception to message
            )
            c = (timeout, {})
        if cache is not None:
            cache[q] = c
    else:
        logger.debug('cache hit')
    t, res = c
    logger.debug('orig query took %.4f s, result:\n%s\n', t, res)
    return t, res


def variable_substitution_query(
        sparql, timeout, graph_pattern, var, source_target_pairs, limit,
        batch_size=config.BATCH_SIZE):
    _vars, _values, _ret_val_mapping = _get_vars_values_mapping(
        graph_pattern, source_target_pairs)
    _sel_var_and_vars = (var, _vars)
    return _multi_query(
        sparql, timeout, graph_pattern, source_target_pairs, batch_size,
        _sel_var_and_vars, _values, _ret_val_mapping,
        _var_subst_res_init, _var_subst_chunk_q, _var_subst_chunk_result_ext,
        limit=limit,  # non standard, passed via **kwds, see handling below
    )


# noinspection PyUnusedLocal
def _var_subst_res_init(_, **kwds):
    return Counter()


def _var_subst_chunk_q(gp, _sel_var_and_vars, values_chunk, limit):
    var, _vars = _sel_var_and_vars
    return gp.to_count_var_over_values_query(
        var=var,
        vars_=_vars,
        values={_vars: values_chunk},
        limit=limit)


# noinspection PyUnusedLocal
def _var_subst_chunk_result_ext(q_res, _sel_var_and_vars, _, **kwds):
    var, _vars = _sel_var_and_vars
    chunk_res = Counter()
    res_rows_path = ['results', 'bindings']
    bindings = sparql_json_result_bindings_to_rdflib(
        get_path(q_res, res_rows_path, default=[])
    )

    for row in bindings:
        row_res = get_path(row, [var])
        count_res = int(get_path(row, [COUNT_VAR], '0'))
        chunk_res[row_res] += count_res
    return chunk_res


def _var_subst_res_update(res, update, **_):
    res += update


def variable_substitution_deep_narrow_mut_query(
        sparql, timeout, graph_pattern, edge_var, node_var,
        source_target_pairs, limit_res, batch_size=config.BATCH_SIZE):
    _vars, _values, _ret_val_mapping = _get_vars_values_mapping(
        graph_pattern, source_target_pairs)
    _edge_var_node_var_and_vars = (edge_var, node_var, _vars)
    return _multi_query(
        sparql, timeout, graph_pattern, source_target_pairs, batch_size,
        _edge_var_node_var_and_vars, _values, _ret_val_mapping,
        _var_subst_dnp_res_init, _var_subst_dnp_chunk_q,
        _var_subst_dnp_chunk_result_ext,
        _res_update=_var_subst_dnp_update,
        limit=limit_res,
        # non standard, passed via **kwds, see handling below
    )


# noinspection PyUnusedLocal
def _var_subst_dnp_res_init(_, **kwds):
    return Counter(), Counter()


def _var_subst_dnp_chunk_q(gp, _edge_var_node_var_and_vars,
                           values_chunk, limit):
    edge_var, node_var, _vars = _edge_var_node_var_and_vars
    return gp.to_find_edge_var_for_narrow_path_query(
        edge_var=edge_var,
        node_var=node_var,
        vars_=_vars,
        values={_vars: values_chunk},
        limit_res=limit)


# noinspection PyUnusedLocal
def _var_subst_dnp_chunk_result_ext(
        q_res, _edge_var_node_var_and_vars, _, **kwds):
    edge_var, node_var, _vars = _edge_var_node_var_and_vars
    chunk_edge_count, chunk_node_sum = Counter(), Counter()
    res_rows_path = ['results', 'bindings']
    bindings = sparql_json_result_bindings_to_rdflib(
        get_path(q_res, res_rows_path, default=[])
    )

    for row in bindings:
        row_res = get_path(row, [edge_var])
        edge_count = int(get_path(row, [EDGE_VAR_COUNT], '0'))
        chunk_edge_count[row_res] += edge_count
        node_sum_count = int(get_path(row, [NODE_VAR_SUM], '0'))
        chunk_node_sum[row_res] += node_sum_count
    return chunk_edge_count, chunk_node_sum,


def _var_subst_dnp_update(res, up, **_):
    edge_count, node_sum_count = res
    try:
        chunk_edge_count, chunk_node_sum = up
        edge_count.update(chunk_edge_count)
        node_sum_count.update(chunk_node_sum)
    except ValueError:
        pass


def generate_stps_from_gp(sparql, gp):
    """Generates a list of source target pairs from a given graph pattern.

    The given graph pattern is immediately used as sparql query to quickly
    generate a list of source target pairs. Possible motivations for this:
    - evaluation: can the algorithm re-discover the given pattern? How does
      complexity of the pattern influence the result / are certain patterns more
      difficult to learn? Think of source target distance for example.
    - completion: one might already know a pattern for connections between
      sources and targets, but not be sure if it's complete. The algorithm can
      be trained on the generated list pairs and might be able to predict
      further targets for given sources.

    :param sparql: SPARQLWrapper endpoint.
    :param gp: GraphPattern to be used as unbound SPARQL select query.
    :return: A list of (source, target) node pairs.
    """
    assert isinstance(gp, GraphPattern)
    q = gp.to_sparql_select_query(projection=(SOURCE_VAR, TARGET_VAR))
    logger.info('generating source target pairs from gp with query:\n%s', q)
    # TODO: continue
