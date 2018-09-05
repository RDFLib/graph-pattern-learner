#!/usr/bin/env python2.7
# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import re
import socket
import sys
from collections import Counter
from collections import OrderedDict
from collections import Sequence
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from datetime import timedelta
from functools import wraps
from time import sleep
from xml.sax.expatreader import SAXParseException

import SPARQLWrapper
import six
from SPARQLWrapper.SPARQLExceptions import EndPointNotFound
from SPARQLWrapper.SPARQLExceptions import SPARQLWrapperException
from cachetools import LRUCache
from rdflib.term import Identifier
# noinspection PyUnresolvedReferences
from six.moves.urllib.error import URLError
from splendid import chunker
from splendid import get_path
from splendid import time_func

import config
from exception import IncompleteQueryException
from exception import MultiQueryException
from exception import MultiQueryClosedException
from graph_pattern import ASK_VAR
from graph_pattern import COUNT_VAR
from graph_pattern import GraphPattern
from graph_pattern import SOURCE_VAR
from graph_pattern import TARGET_VAR
from utils import exception_stack_catcher
from utils import kv_str
from utils import sparql_json_result_bindings_to_rdflib
from utils import timer

logger = logging.getLogger(__name__)


class _QueryStats(object):
    def __init__(self):
        self.queries = 0
        self.query_cache_hits = 0
        self.query_cache_misses = 0

        self.multi_query_count = Counter()
        self.multi_query_chunks = Counter()
        self.multi_query_retries = Counter()
        self.multi_query_splits = Counter()

        self.ask_multi_query_count = 0
        self.combined_ask_count_multi_query_count = 0
        self.variable_substitution_query_count = 0
        self.useful_path_query_count = 0
        self.useful_path_inst_query_count = 0
        self.predict_query_count = 0
        self.count_query_count = 0

        self.guard = None


    def __str__(self):
        above_0 = ""
        if self.count_query_count:
            above_0 += "\n    count_query: %d" % self.count_query_count
        if self.predict_query_count:
            above_0 += "\n    predict_query: %d" % self.predict_query_count
        return (
            "  Queries: %d total, cache: %d hits, %d misses\n"
            "  Multi-Query:\n"
            "    count: %d, batch sizes: %s\n"
            "    chunks: %d, batch sizes: %s\n"
            "    retries: %d, batch sizes: %s\n"
            "    splits: %d, batch sizes: %s\n"
            "  High-Level query functions:\n"
            "    ask_multi_query: %d\n"
            "    combined_ask_count_multi_query: %d\n"
            "    variable_substitution_query: %d%s"
            % (
                self.queries, self.query_cache_hits, self.query_cache_misses,
                sum(self.multi_query_count.values()),
                kv_str(self.multi_query_count.most_common()),
                sum(self.multi_query_chunks.values()),
                kv_str(self.multi_query_chunks.most_common()),
                sum(self.multi_query_retries.values()),
                kv_str(self.multi_query_retries.most_common()),
                sum(self.multi_query_splits.values()),
                kv_str(self.multi_query_splits.most_common()),
                self.ask_multi_query_count,
                self.combined_ask_count_multi_query_count,
                self.variable_substitution_query_count,
                above_0,
            )
        )

    def __add__(self, other):
        assert isinstance(other, _QueryStats)
        res = _QueryStats()
        for k, vs in vars(self).items():
            if k == 'guard':
                # keep guard as is
                setattr(res, k, vs)
            else:
                # sum all else
                setattr(res, k, vs + getattr(other, k))
        return res

    def __radd__(self, other):
        if other == 0:
            # allow simple sum()
            return self

    def __sub__(self, other):
        assert isinstance(other, _QueryStats)
        res = _QueryStats()
        for k, vs in vars(self).items():
            if k == 'guard':
                # keep guard as is
                setattr(res, k, vs)
            else:
                # sum all else
                setattr(res, k, vs - getattr(other, k))
        return res


@exception_stack_catcher
def query_stats(guard):
    global _query_stats_last_adapt
    if _query_stats.guard != guard:
        _query_stats.guard = guard
        bs = config.BATCH_SIZE
        logger.debug('QueryStats:\n%s' % _query_stats)

        if config.BATCH_SIZE_ADAPT:
            # adapt batch size if necessary
            diff = _query_stats - _query_stats_last_adapt
            splits = diff.multi_query_splits[config.BATCH_SIZE]
            queries = diff.multi_query_count[config.BATCH_SIZE]
            if splits > .1 * queries:
                # > 10 % of orig queries since last adapt resulted in splits
                if config.BATCH_SIZE > config.BATCH_SIZE_MIN:
                    config.BATCH_SIZE = max(
                        int(config.BATCH_SIZE * .75),
                        config.BATCH_SIZE_MIN
                    )
                    logger.warning(
                        'too many splits, reduced future BATCH_SIZE to %d, '
                        'consider restarting learning with --BATCH_SIZE=%d',
                        config.BATCH_SIZE, config.BATCH_SIZE
                    )
                else:
                    logger.warning(
                        'too many splits even with MIN_BATCH_SIZE=%d, is '
                        'something wrong with the endpoint or the URIs?',
                        config.BATCH_SIZE
                    )
                _query_stats_last_adapt = deepcopy(_query_stats)
        return _query_stats, bs
_query_stats = _QueryStats()
_query_stats_last_adapt = _QueryStats()




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
            raise IncompleteQueryException(
                "tried to run a query on a graph pattern without "
                "%s and %s vars:\n%s" % (SOURCE_VAR, TARGET_VAR, graph_pattern)
            )
        # remember which source_target_pair is reached via which value
        for stp in source_target_pairs:
            _ret_val_mapping[(stp[_val_idx],)].append(stp)
    return _vars, _values, _ret_val_mapping


def ask_multi_query(
        sparql, timeout, graph_pattern, source_target_pairs,
        batch_size=None):
    assert isinstance(source_target_pairs, Sequence)
    assert isinstance(source_target_pairs[0], tuple)
    _query_stats.ask_multi_query_count += 1
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


def _exception_closes_worker_guard(func):
    """Temporarily closes _multi_query for current worker.

    This is a workaround for SCOOP's map otherwise having already dispatched
    further work to this worker, despite an exception of a previous _multi_query
    not being handled in origin yet.

    An exception being raised out of _multi_query would normally cause origin to
    back-off for config.ERROR_WAIT and retry. This "quick fails" all remaining
    work in the time frame.
    """
    closed = []
    wait = timedelta(
        seconds=config.ERROR_WAIT * .75  # rather don't close too long
    )

    @wraps(func)
    def _multi_query_wrapper(*args, **kwds):
        if closed:
            if datetime.utcnow() - closed[0] < wait:
                logger.warning(
                    '_multi_query temporarily closed for worker due to '
                    'previous exception'
                )
                raise MultiQueryClosedException('closed for worker')
            else:
                closed.pop()
        try:
            return func(*args, **kwds)
        except:
            closed.append(datetime.utcnow())
            raise
    return _multi_query_wrapper


@_exception_closes_worker_guard
def _multi_query(
        sparql, timeout, graph_pattern, source_target_pairs,
        batch_size,
        _vars, _values, _ret_val_mapping,
        _res_init, _chunk_q, _chunk_res,
        _res_update=lambda r, u, **___: r.update(u),
        **kwds):
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    _query_stats.multi_query_count[batch_size] += 1
    total_time = 0
    res = _res_init(source_target_pairs, **kwds)
    for val_chunk in chunker(_values, batch_size):
        _query_stats.multi_query_chunks[batch_size] += 1
        q = _chunk_q(graph_pattern, _vars, val_chunk, **kwds)
        chunk_stps = [stp for v in val_chunk for stp in _ret_val_mapping[v]]
        _start_time = timer()
        t = None
        chunk_res = None
        for retry in range(2, -1, -1):  # 3 attempts: 2, 1, 0
            if retry < 2:
                _query_stats.multi_query_retries[batch_size] += 1
            try:
                t, q_res = _query(sparql, timeout, q, **kwds)
                chunk_res = _chunk_res(
                    q_res, _vars, _ret_val_mapping, **kwds)
            except EndPointNotFound as e:
                # happens if the endpoint reports a 404...
                # as virtuoso in rare cases seems to report a 404 let's
                # retry after some time but then cancel
                if retry:
                    logger.info(
                        'SPARQL endpoint reports a 404, will retry in %ds',
                        config.ERROR_WAIT
                    )
                    sleep(config.ERROR_WAIT)
                    continue
                else:
                    logger.exception(
                        'SPARQL endpoint unreachable even after back-off '
                        'and retry\n'
                        'could not perform query:\n%s for %s\nException:',
                        q, val_chunk,
                    )
                    six.reraise(MultiQueryException, e, sys.exc_info()[2])
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
                    _query_stats.multi_query_splits[batch_size] += 1
                    t, chunk_res = _multi_query(
                        sparql, timeout, graph_pattern, chunk_stps,
                        len(val_chunk) // 2,
                        _vars, val_chunk, _ret_val_mapping,
                        _res_init, _chunk_q, _chunk_res,
                        _res_update,
                        **kwds)
                elif isinstance(e, URLError):
                    # we're down at single query level and still encounter an
                    # error. It is very likely that the endpoint is dead...
                    if retry:
                        logger.warning(
                            'URLError, seems we cannot reach SPARQL endpoint, '
                            'retry in %ds. Tried to perform query:\n'
                            '%s for %s\nException:',
                            config.ERROR_WAIT, q, val_chunk,
                            exc_info=1,  # appends exception to message
                        )
                        sleep(config.ERROR_WAIT)
                        continue
                    else:
                        logger.exception(
                            'URLError, seems we cannot reach SPARQL endpoint, '
                            'giving up after 3 retries. Tried to perform query:'
                            '\n%s for %s\nException:',
                            q, val_chunk,
                        )
                        six.reraise(MultiQueryException, e, sys.exc_info()[2])
                else:
                    logger.warning(
                        'could not perform query, replacing with 0 result:\n'
                        '%s for %s\nException:',
                        q, val_chunk,
                        exc_info=1,  # appends exception to message
                    )
                    t, chunk_res = timer() - _start_time, {}
            except Exception as e:
                if retry:
                    logger.warning(
                        'unhandled exception, retry in %ds:\n'
                        'Query:\n%s\nChunk:%r\nException:',
                        config.ERROR_WAIT, q, val_chunk,
                        exc_info=1,  # appends exception to message
                    )
                    sleep(config.ERROR_WAIT)
                    continue
                else:
                    logger.exception(
                        'unhandled exception, giving up after 3 retries:\n'
                        'Query:\n%s\nChunk:%r\nException:',
                        q, val_chunk,
                    )
                    six.reraise(MultiQueryException, e, sys.exc_info()[2])
            break
        _res_update(res, chunk_res, **kwds)
        total_time += t
        if query_time_soft_exceeded(total_time, timeout):
            logger.debug('early terminating batch query as timeout/2 exceeded')
            break
    return total_time, res


def combined_ask_count_multi_query(
        sparql, timeout, graph_pattern, source_target_pairs,
        batch_size=None):
    _query_stats.combined_ask_count_multi_query_count += 1
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
    _query_stats.count_query_count += 1

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
def predict_multi_query(
        sparql, timeout, graph_pattern, sources,
        batch_size=None):
    assert isinstance(graph_pattern, GraphPattern)
    assert graph_pattern.complete()
    t, res = _multi_query(
        sparql, timeout, graph_pattern, sources, batch_size,
        (SOURCE_VAR,), [(s,) for s in sources], {(s,): (s,) for s in sources},
        _predict_res_init, _predict_chunk_q, _predict_chunk_res,
    )
    return t, OrderedDict([(s, set(tcs)) for s, tcs in res.items()])


def _predict_res_init(sources):
    return OrderedDict([(s, []) for s in sources])


def _predict_chunk_q(gp, _vars, values_chunk):
    assert isinstance(gp, GraphPattern)
    return gp.to_sparql_select_query(
        projection=[SOURCE_VAR, TARGET_VAR],
        distinct=True,
        values={_vars: values_chunk},
        limit=config.PREDICTION_RESULT_LIMIT,
    )


def _predict_chunk_res(q_res, *_):
    chunk_res = defaultdict(list)
    res_rows_path = ['results', 'bindings']
    bindings = sparql_json_result_bindings_to_rdflib(
        get_path(q_res, res_rows_path, default=[])
    )
    for row in bindings:
        s = get_path(row, [SOURCE_VAR])
        t = get_path(row, [TARGET_VAR])
        chunk_res[s].append(t)
    return chunk_res


@exception_stack_catcher
def predict_query(sparql, timeout, graph_pattern, source,
                  limit=config.PREDICTION_RESULT_LIMIT):
    """Performs a single query starting at ?SOURCE returning all ?TARGETs."""
    assert isinstance(graph_pattern, GraphPattern)
    assert isinstance(source, Identifier)
    _query_stats.predict_query_count += 1

    vars_in_graph = graph_pattern.vars_in_graph
    if TARGET_VAR not in vars_in_graph:
        logger.warning(
            'graph pattern without %s used for prediction:\n%r',
            TARGET_VAR.n3(), graph_pattern
        )
        return timeout, set()

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
    _query_stats.queries += 1
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
        _query_stats.query_cache_misses += 1
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
        _query_stats.query_cache_hits += 1
    t, res = c
    logger.debug('orig query took %.4f s, result:\n%s\n', t, res)
    return t, res


def variable_substitution_query(
        sparql, timeout, graph_pattern, var, source_target_pairs, limit,
        batch_size=None):
    _query_stats.variable_substitution_query_count += 1
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
    

def deep_narrow_path_query(
        sparql,
        timeout,
        graph_pattern,
        var_to_fix,
        var_to_count,
        valueblocks,
        steps,
        startvar,
        avglimit=config.MUTPB_DN_AVG_DEG_LIMIT,
        gp_in=False,
        batch_size=None
):
    _query_stats.useful_path_query_count += 1
    # TODO: evtl. je 10 pro 'gefixter' Variable von batch-size abziehen
    # (weil der Block ja mit rein geht)
    _values = graph_pattern.matching_node_pairs
    # TODO: evtl. Schnitt mit noch nicht abgedeckten
    _ret_val_mapping = {stp: [stp] for stp in graph_pattern.matching_node_pairs}
    _vars_steps_and_stuff = (
        var_to_fix, var_to_count, startvar, valueblocks, steps, avglimit, gp_in
    )
    return _multi_query(
        sparql, timeout, graph_pattern, graph_pattern.matching_node_pairs,
        batch_size, _vars_steps_and_stuff, _values, _ret_val_mapping,
        _deep_narrow_path_res_init, _deep_narrow_path_chunk_q,
        _deep_narrow_path_chunk_result_ext, _deep_narrow_path_res_update
    )


# noinspection PyUnusedLocal
def _deep_narrow_path_res_init(_, **kwds):
    return []


def _deep_narrow_path_chunk_q(gp, _vars_steps_and_stuff, values_chunk):
    var_to_fix, var_to_count, startvar, _valueblocks, steps, avglimit, gp_in \
        = _vars_steps_and_stuff
    valueblocks = {
        startvar: {
            (startvar,):
                [(tup[0],) for tup in values_chunk] if startvar == SOURCE_VAR
                else [(tup[1],) for tup in values_chunk]
        }
    }
    valueblocks.update(_valueblocks)
    return gp.to_sparql_deep_narrow_path_query(
            var_to_fix,
            var_to_count,
            valueblocks,
            steps,
            startvar,
            avglimit=avglimit,
            gp_in=gp_in
    )


# noinspection PyUnusedLocal
def _deep_narrow_path_chunk_result_ext(q_res, _vars_steps_and_stuff, _, **kwds):
    var_to_fix, var_to_count, startvar, _valueblocks, steps, avglimit, gp_in \
        = _vars_steps_and_stuff
    chunk_res = []
    res_rows_path = ['results', 'bindings']
    bindings = sparql_json_result_bindings_to_rdflib(
        get_path(q_res, res_rows_path, default=[])
    )

    for row in bindings:
        # TODO: Drüber nachdenken, ob iwie die avg-outgoing auch mit
        # zurückgegeben werden sollen
        chunk_res.append(get_path(row, [var_to_fix]))
    return chunk_res


def _deep_narrow_path_res_update(res, update, **_):
    res += update
    
    
def deep_narrow_path_inst_query(
        sparql,
        timeout,
        graph_pattern,
        hop,
        valueblocks,
        steps,
        gp_in=False,
        batch_size=None
):
    _query_stats.useful_path_inst_query_count += 1
    # TODO: evtl. je 10 pro 'gefixter' Variable von batch-size abziehen
    # (weil der Block ja mit rein geht)
    _values = graph_pattern.matching_node_pairs
    # evtl. Schnitt mit noch nicht abgedeckten
    _ret_val_mapping = {stp: [stp] for stp in graph_pattern.matching_node_pairs}
    _vars_steps_and_stuff = (hop, valueblocks, steps, gp_in)
    return _multi_query(
        sparql, timeout, graph_pattern, graph_pattern.matching_node_pairs,
        batch_size, _vars_steps_and_stuff, _values, _ret_val_mapping,
        _deep_narrow_path_inst_res_init, _deep_narrow_path_inst_chunk_q,
        _deep_narrow_path_inst_chunk_result_ext,
        _deep_narrow_path_inst_res_update
    )


# noinspection PyUnusedLocal
def _deep_narrow_path_inst_res_init(_, **kwds):
    return []


def _deep_narrow_path_inst_chunk_q(gp, _vars_steps_and_stuff, values_chunk):
    hop, _valueblocks, steps, gp_in = _vars_steps_and_stuff
    valueblocks = {
        'st': {
            (SOURCE_VAR, TARGET_VAR): values_chunk
        }
    }
    valueblocks.update(_valueblocks)
    return gp.to_sparql_deep_narrow_path_inst_query(
        hop, valueblocks, steps, gp_in=gp_in
    )


# noinspection PyUnusedLocal
def _deep_narrow_path_inst_chunk_result_ext(
        q_res, _vars_steps_and_stuff, _, **kwds
):
    hop, _valueblocks, steps, gp_in = _vars_steps_and_stuff
    chunk_res = []
    res_rows_path = ['results', 'bindings']
    bindings = sparql_json_result_bindings_to_rdflib(
        get_path(q_res, res_rows_path, default=[])
    )

    for row in bindings:
        chunk_res.append([get_path(row, [h]) for h in hop])
    return chunk_res


def _deep_narrow_path_inst_res_update(res, update, **_):
    res += update


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
