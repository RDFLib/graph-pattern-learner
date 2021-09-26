#!/usr/bin/env python2.7
# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
if __name__ == '__main__':
    print(
        "You probably wanted to execute run.py instead of this.",
        file=sys.stderr
    )
    sys.exit(1)

from collections import Counter
from collections import defaultdict
from collections import OrderedDict
from functools import partial
from operator import attrgetter
import random
import signal
from time import sleep
from datetime import datetime

from cachetools.func import lru_cache
import deap
import deap.base
import deap.tools
import numpy as np
from rdflib import BNode
from rdflib import Literal
from rdflib import URIRef
from rdflib import Variable
from rdflib import XSD
import SPARQLWrapper
from scoop.futures import map as parallel_map
import six

import logging
logger = logging.getLogger(__name__)

import logging_config
from cluster import expected_precision_loss_by_query_reduction
from cluster import cluster_gps_to_reduce_queries
import config
from exception import GPLearnerAbortException
from fusion import fuse_prediction_results
from fusion import train_fusion_models
from gp_query import ask_multi_query
from gp_query import calibrate_query_timeout
from gp_query import combined_ask_count_multi_query
from gp_query import predict_multi_query
from gp_query import predict_query
from gp_query import query_stats
from gp_query import query_time_hard_exceeded
from gp_query import query_time_soft_exceeded
from gp_query import deep_narrow_path_query
from gp_query import deep_narrow_path_inst_query
from gp_query import variable_substitution_query
from graph_pattern import canonicalize
from graph_pattern import gen_random_var
from graph_pattern import GPFitness
from graph_pattern import GPFitnessTuple
from graph_pattern import GraphPattern
from graph_pattern import GraphPatternStats
from graph_pattern import replace_vars_with_random_vars
from graph_pattern import SOURCE_VAR
from graph_pattern import TARGET_VAR
from ground_truth_tools import get_semantic_associations
from ground_truth_tools import k_fold_cross_validation
from ground_truth_tools import split_training_test_set
from gtp_scores import GTPScores
from memory_usage import log_mem_usage
from serialization import find_last_result
from serialization import load_predicted_target_candidates
from serialization import save_predicted_target_candidates
from serialization import find_run_result
from serialization import format_graph_pattern
from serialization import load_init_patterns
from serialization import load_results
from serialization import pause_if_signaled_by_file
from serialization import print_graph_pattern
from serialization import print_population
from serialization import print_results
from serialization import remove_old_result_files
from serialization import save_generation
from serialization import save_results
from serialization import save_run
from serialization import set_symlink
from utils import exception_stack_catcher
from utils import kv_str
from utils import log_all_exceptions
from utils import log_wrapped_exception
from utils import sample_from_list


logger.info('init gp_learner')
signal.signal(signal.SIGUSR1, log_mem_usage)


def init_workers():
    parallel_map(_init_workers, range(1000))


def _init_workers(_):
    # dummy method that makes workers load all import and config
    pass


def f_measure(precision, recall, beta=config.F_MEASURE_BETA):
    """Calculates the f1-measure from precision and recall."""
    if precision + recall <= 0:
        return 0.
    beta_sq = beta ** 2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


@exception_stack_catcher
def evaluate(sparql, timeout, gtp_scores, graph_pattern, run=0, gen=0):
    assert isinstance(graph_pattern, GraphPattern)
    assert isinstance(gtp_scores, GTPScores)
    ground_truth_pairs = gtp_scores.ground_truth_pairs
    remaining_gain = gtp_scores.remaining_gain
    gtp_max_precisions = gtp_scores.gtp_max_precisions

    complete_pattern = 1 if graph_pattern.complete() else 0
    pattern_length = len(graph_pattern)
    vars_in_graph_pattern = graph_pattern.vars_in_graph
    pattern_vars = len(vars_in_graph_pattern)

    logger.debug('evaluating %s', graph_pattern)

    # check how many gt_matches (recall) and targets (precision) can be reached
    # there are several cases here:
    # - ?source & ?target in pattern:
    #   we can use a combined_ask_count_multi_query to only use one query and
    #   get gt_matches and res_lengths at once
    # - only ?source in pattern
    #   we can check how many gt_matches (recall) there are with ask_multi_query
    #   but as we don't have a ?target variable we can't ever reach a target
    #   with this pattern, meaning we have res_lengths = [] --> precision = 0
    # - only ?target in pattern
    #   we can check how many gt_matches (recall) there are with ask_multi_query
    #   we could still get res_lengths with a count_query, which might be useful
    #   for universal targets? 'select count(?target) as ?count' but it turns
    #   out this is bad idea:
    #   - it gives incomplete ?target var patterns an advantage against
    #     ?source only patterns (which will always have a precision of 0)
    #   - the precision of a ?target var only pattern is preferring very narrow
    #     patterns (one could say over-fitting) which are only tailored towards
    #     singling out one specific ?target variable as they can't rely on a
    #     ?source to actually help with the filtering. Example:
    #     (dbpedia-de:Pferd owl:sameAs ?target) (only returns dbpedia:Horse)
    # - none of the above in pattern: should never happen
    if complete_pattern:
        query_time, (gt_matches, res_lengths) = combined_ask_count_multi_query(
            sparql, timeout, graph_pattern, ground_truth_pairs)
    else:
        # run an ask_multi_query to see how many gt_matches (recall) we have
        query_time, gt_matches = ask_multi_query(
            sparql, timeout, graph_pattern, ground_truth_pairs)
        res_lengths = {gtp: 0 for gtp in ground_truth_pairs}

    matching_node_pairs = [
        gtp for gtp in ground_truth_pairs if gt_matches[gtp]
    ]

    # TODO: maybe punish patterns which produce results but don't match?

    # judging query times:
    # We can't trust counts returned by timed out queries. Nevertheless, we want
    # to give soft-timeout queries a slight advantage over complete failures.
    # For this we use the f_measure component which still gets a severe
    # punishment below, while the individual counts and via that gain and score
    # are ignored in case of a timeout.
    qtime_exceeded = 0
    if query_time_hard_exceeded(query_time, timeout):
        qtime_exceeded = 1
    elif query_time_soft_exceeded(query_time, timeout):
        qtime_exceeded = .5
    trust = (1 - qtime_exceeded)

    sum_gt_matches = sum(gt_matches.values())
    recall = sum_gt_matches / len(ground_truth_pairs)
    non_zero_res_lens = [l for l in res_lengths.values() if l > 0]
    avg_res_len = sum(non_zero_res_lens) / max(len(non_zero_res_lens), 1)
    precision = 0
    if avg_res_len > 0:
        precision = 1 / avg_res_len
    fm = f_measure(precision, recall) * trust

    gtp_precisions = OrderedDict()
    gain = 0
    if qtime_exceeded == 0:
        for gtp in matching_node_pairs:
            gtp_res_len = res_lengths[gtp]
            if gtp_res_len > 0:
                gtp_precision = 1 / gtp_res_len
                gtp_precisions[gtp] = gtp_precision
                score_diff = gtp_precision - gtp_max_precisions[gtp]
                if score_diff > 0:
                    gain += score_diff

    # counter overfitting
    overfitting = 1
    if matching_node_pairs:
        m_sources, m_targets = zip(*matching_node_pairs)
        if len(set(m_sources)) < 2:
            overfitting *= config.OVERFITTING_PUNISHMENT
        if len(set(m_targets)) < 2:
            overfitting *= config.OVERFITTING_PUNISHMENT

    score = trust * overfitting * gain

    # order of res needs to fit to graph_pattern.GPFitness
    res = (
        remaining_gain,
        score,
        gain,
        fm,
        avg_res_len,
        sum_gt_matches,
        pattern_length,
        pattern_vars,
        qtime_exceeded,
        query_time,
    )
    logger.log(
        config.LOGLVL_EVAL,
        'Run %d, Generation %d: evaluated fitness for %s%s\n%s',
        run, gen,
        graph_pattern,
        GPFitness(res).format_fitness(),
        graph_pattern.fitness.description
    )
    return res, matching_node_pairs, gtp_precisions


def update_individuals(individuals, eval_results):
    """Updates the given individuals with the eval_results in-place.

    :param individuals: A list of individuals (GraphPatterns).
    :param eval_results: a list of results calculated by evaluate(), in the
        order of individuals.
    :return: None
    """
    for ind, res in zip(individuals, eval_results):
        ind.fitness.values = res[0]
        ind.matching_node_pairs = res[1]
        ind.gtp_precisions = res[2]


@lru_cache(maxsize=config.CACHE_SIZE_FIT_TO_LIVE)
def fit_to_live(child):
    if 1 > len(child) > config.MAX_PATTERN_LENGTH:
        return False
    if not child.vars_in_graph & {SOURCE_VAR, TARGET_VAR}:
        return False
    if len(child.vars_in_graph) > config.MAX_PATTERN_VARS:
        return False
    if len(child.to_sparql_select_query()) > config.MAX_PATTERN_QUERY_SIZE:
        return False
    if any([
        len(o.n3()) > config.MAX_LITERAL_SIZE
        for _, _, o in child if isinstance(o, Literal)
    ]):
        return False
    if any([
        isinstance(s, Literal) or isinstance(p, (BNode, Literal))
        for s, p, _ in child
    ]):
        return False
    if not child.is_connected(via_edges=config.PATTERN_P_CONNECTED):
        return False
    return True


def mate_helper(
        overlap,
        delta_dom,
        delta_other,
        pb_overlap,
        pb_dom,
        pb_other,
        pb_rename_delta_vars,
        retries,
):
    assert isinstance(overlap, set)
    assert isinstance(delta_dom, set)
    assert isinstance(delta_other, set)
    for _ in range(retries):
        overlap_part = [t for t in overlap if random.random() < pb_overlap]
        dom_part = [t for t in delta_dom if random.random() < pb_dom]
        other_part = [t for t in delta_other if random.random() < pb_other]
        if random.random() < pb_rename_delta_vars:
            other_part = replace_vars_with_random_vars(other_part)
        child = canonicalize(GraphPattern(overlap_part + dom_part + other_part))
        if fit_to_live(child):
            return child
        else:
            # most likely not connected, try connecting by merging vars nodes
            child = canonicalize(mutate_merge_var(child))
            if fit_to_live(child):
                return child
    return None


def mate(
        individual1,
        individual2,
        pb_overlap=config.CXPB_BP,
        pb_dominant_parent=config.CXPB_DP,
        pb_other_parent=config.CXPB_OP,
        pb_rename_delta_vars=config.CXPB_RV,
        retries=config.CX_RETRY,
):
    # mate patterns:
    # we return 2 children:
    # child1 who's dominant parent is individual1
    # child2 who's dominant parent is individual2
    # both children draw triples from the overlap with pb_overlap.
    # child1 draws each triple of individual1 with prob pb_dominant_parent and
    # each triple of individual2 with prob pb_other_parent.
    # child2 swaps the probabilities accordingly.
    # the drawings are repeated retries times. If no fit_to_live child is found
    # the dominant parent is returned
    assert fit_to_live(individual1), 'unfit indiv in mating %r' % (individual1,)
    assert fit_to_live(individual2), 'unfit indiv in mating %r' % (individual2,)
    overlap = set(individual1) & set(individual2)
    delta1 = set(individual1) - overlap
    delta2 = set(individual2) - overlap
    child1 = mate_helper(
        overlap, delta1, delta2,
        pb_overlap, pb_dominant_parent, pb_other_parent, pb_rename_delta_vars,
        retries,
    ) or individual1
    child2 = mate_helper(
        overlap, delta2, delta1,
        pb_overlap, pb_dominant_parent, pb_other_parent, pb_rename_delta_vars,
        retries
    ) or individual2
    assert fit_to_live(child1), 'mating %r and %r produced unfit child %r' % (
        individual1, individual2, child1
    )
    assert fit_to_live(child2), 'mating %r and %r produced unfit child %r' % (
        individual1, individual2, child2
    )
    return child1, child2


def mutate_introduce_var(child):
    identifiers = tuple(child.identifier_counts(exclude_vars=True))
    if not identifiers:
        return child
    identifier = random.choice(identifiers)
    rand_var = gen_random_var()
    return GraphPattern(child, mapping={identifier: rand_var})


def mutate_split_var(child):
    # count triples that each var occurs in (not occurrences: (?s ?p ?s))
    var_trip_count = Counter([
        ti for t in child for ti in set(t) if isinstance(ti, Variable)
        # if ti not in (SOURCE_VAR, TARGET_VAR)  # why not allow them to split?
    ])
    # select vars that occur multiple times
    var_trip_count = Counter({i: c for i, c in var_trip_count.items() if c > 1})
    if not var_trip_count:
        return child
    var_to_split = random.choice(list(var_trip_count.elements()))
    triples_with_var = [t for t in child if var_to_split in t]
    triples = [t for t in child if var_to_split not in t]

    # randomly split triples_with_var into 2 non-zero length parts:
    # the first part where var_to_split is substituted and the 2nd where not
    random.shuffle(triples_with_var)
    split_idx = random.randrange(1, len(triples_with_var))
    triples += triples_with_var[split_idx:]
    rand_var = gen_random_var()
    triples += [
        tuple([rand_var if ti == var_to_split else ti for ti in t])
        for t in triples_with_var[:split_idx]
    ]
    gp = GraphPattern(triples)
    if not fit_to_live(gp):
        # can happen that we created a disconnected pattern:
        # orig:
        #  ?s ?p ?X, ?X ?q ?Y, ?Y ?r ?t
        # splitvar X:
        #  ?s ?p ?Z, ?X ?q ?Y, ?Y ?r ?t
        # try merging once, might lead to this:
        #  ?s ?p ?Y, ?X ?q ?Y, ?Y ?r ?t
        gp = mutate_merge_var(gp)
        if not fit_to_live(gp):
            return child
    return gp


def mutate_merge_var(child, pb_mv_mix=config.MUTPB_MV_MIX):
    if random.random() < pb_mv_mix:
        return mutate_merge_var_mix(child)
    else:
        return mutate_merge_var_sep(child)


def _mutate_merge_var_helper(vars_):
    rand_vars = vars_ - {SOURCE_VAR, TARGET_VAR}
    merge_able_vars = len(rand_vars) - 1
    if len(vars_) > len(rand_vars):
        # either SOURCE_VAR or TARGET_VAR is also available as merge target
        merge_able_vars += 1
    merge_able_vars = max(0, merge_able_vars)
    return rand_vars, merge_able_vars


def mutate_merge_var_mix(child):
    """Merges two variables into one, potentially merging node and edge vars."""
    vars_ = child.vars_in_graph
    rand_vars, merge_able_vars = _mutate_merge_var_helper(vars_)

    if merge_able_vars < 1:
        return child

    # merge vars, even mixing nodes and edges
    var_to_replace = random.choice(list(rand_vars))
    var_to_merge_into = random.choice(list(vars_ - {var_to_replace}))
    return GraphPattern(child, mapping={var_to_replace: var_to_merge_into})


def mutate_merge_var_sep(child):
    """Merges two variables into one, won't merge node and edge vars.

    Considers the node variables and edge variables separately.
    Depending on availability either merges 2 node variables or 2 edge variable.
    """
    node_vars = {n for n in child.nodes if isinstance(n, Variable)}
    rand_node_vars, merge_able_node_vars = _mutate_merge_var_helper(node_vars)

    edge_vars = {e for e in child.edges if isinstance(e, Variable)}
    rand_edge_vars, merge_able_edge_vars = _mutate_merge_var_helper(edge_vars)

    if merge_able_node_vars < 1 and merge_able_edge_vars < 1:
        return child

    # randomly merge node or predicate vars proportional to their occurrences
    r = random.randrange(0, merge_able_node_vars + merge_able_edge_vars)
    if r < merge_able_node_vars:
        # we're merging node vars
        var_to_replace = random.choice(list(rand_node_vars))
        var_to_merge_into = random.choice(list(
            node_vars - {var_to_replace}))
    else:
        # we're merging predicate vars
        var_to_replace = random.choice(list(rand_edge_vars))
        var_to_merge_into = random.choice(list(
            edge_vars - {var_to_replace}))

    return GraphPattern(child, mapping={var_to_replace: var_to_merge_into})


def mutate_del_triple(child):
    l = len(child)
    if l < 2:
        return child
    new_child = GraphPattern(random.sample(child, l - 1))
    if not fit_to_live(new_child):
        return child
    else:
        return new_child


def _mutate_expand_node_helper(node, pb_en_out_link=config.MUTPB_EN_OUT_LINK):
    """Adds a new var-only triple to node.

    :param pb_en_out_link: Probability to create an outgoing triple.
    :return: The new triple, node and var
    """
    var_edge = gen_random_var()
    var_node = gen_random_var()
    if random.random() < pb_en_out_link:
        new_triple = (node, var_edge, var_node)
    else:
        new_triple = (var_node, var_edge, node)
    return new_triple, var_node, var_edge


def mutate_expand_node(
        child, node=None, pb_en_out_link=config.MUTPB_EN_OUT_LINK):
    """Expands a random node by adding a new var-only triple to it.

    Randomly selects a node. Then adds an outgoing or incoming triple with two
    new vars to it.

    :param child: The GraphPattern to expand a node in.
    :param node: If given the node to expand, otherwise
    :param pb_en_out_link: Probability to create an outgoing triple.
    :return: A child with the added outgoing/incoming triple.
    """
    # TODO: can maybe be improved by sparqling
    if not node:
        nodes = list(child.nodes)
        node = random.choice(nodes)
    new_triple, _, _ = _mutate_expand_node_helper(node, pb_en_out_link)
    return child + (new_triple,)


def mutate_add_edge(child):
    """Adds an edge between 2 randomly selected nodes.

    Randomly selects two nodes, then adds a new triple (n1, e, n2), where e is
    a new variable.

    :return: A child with the added edge.
    """
    # TODO: can maybe be improved by sparqling
    nodes = list(child.nodes)
    if len(nodes) < 2:
        return child
    node1, node2 = random.sample(nodes, 2)
    var_edge = gen_random_var()
    new_triple = (node1, var_edge, node2)
    return child + (new_triple,)


def mutate_increase_dist(child):
    """Increases the distance between ?source and ?target by one hop.

    Randomly adds a var only triple to the ?source or ?target var. Then swaps
    the new node with ?source/?target to increase the distance by one hop.

    :return: A child with increased distance between ?source and ?target.
    """
    if not child.complete():
        return child
    var_node = gen_random_var()
    var_edge = gen_random_var()
    old_st = random.choice([SOURCE_VAR, TARGET_VAR])
    new_triple = random.choice([
        (old_st, var_edge, var_node),  # outgoing new triple
        (var_node, var_edge, old_st),  # incoming new triple
    ])
    new_child = child + (new_triple,)
    # replace the old source/target node with the new node and vice-versa to
    # move the old node one hop further away from everything else
    new_child = new_child.replace({old_st: var_node, var_node: old_st})
    return new_child


def mutate_fix_var_filter(item_counts):
    """Filters results of fix var mutation in-place.

    Excludes:
    - too long literals
    - URIs with encoding errors (real world!)
    - BNode results (they will not be fixed but stay SPARQL vars)
    - NaN or INF literals (Virtuoso bug
        https://github.com/openlink/virtuoso-opensource/issues/649 )
    """
    assert isinstance(item_counts, Counter)
    for i in list(item_counts.keys()):
        if isinstance(i, Literal):
            i_n3 = i.n3()
            if len(i_n3) > config.MAX_LITERAL_SIZE:
                logger.debug(
                    'excluding very long literal %d > %d from mutate_fix_var:\n'
                    '%s...',
                    len(i_n3), config.MAX_LITERAL_SIZE, i_n3[:128]
                )
                del item_counts[i]
            elif i.datatype in (XSD['float'], XSD['double']) \
                    and six.text_type(i).lower() in ('nan', 'inf'):
                logger.debug('excluding %s due to Virtuoso Bug', i_n3)
                del item_counts[i]
        elif isinstance(i, URIRef):
            # noinspection PyBroadException
            try:
                i.n3()
            except Exception:  # sadly RDFLib doesn't raise a more specific one
                # it seems some SPARQL endpoints (Virtuoso) are quite liberal
                # during their import process, so it can happen that we're
                # served broken URIs, which break when re-inserted into SPARQL
                # later by calling URIRef.n3()
                logger.warning(
                    'removed invalid URI from mutate_fix_var:\n%r',
                    i
                )
                del item_counts[i]
        elif isinstance(i, BNode):
            # make sure that BNodes stay variables
            logger.info('removed BNode from mutate_fix_var')
            del item_counts[i]
        else:
            logger.warning(
                'exlcuding unknown result type from mutate_fix_var:\n%r',
                i
            )
            del item_counts[i]


@exception_stack_catcher
def mutate_fix_var(
        sparql,
        timeout,
        gtp_scores,
        child,
        gtp_sample_max_n=config.MUTPB_FV_RGTP_SAMPLE_N,
        rand_var=None,
        sample_max_n=config.MUTPB_FV_SAMPLE_MAXN,
        limit=config.MUTPB_FV_QUERY_LIMIT,
):
    """Finds possible fixations for a randomly selected variable of the pattern.

    This is the a very important mutation of the gp learner, as it is the main
    source of actually gaining information from the SPARQL endpoint.

    The outline of the mutation is as follows:
    - If not passed in, randomly selects a variable (rand_var) of the pattern
      (node or edge var, excluding ?source and ?target).
    - Randomly selects a subset of up to gtp_sample_max_n GTPs with
      probabilities according to their remaining gains. The number of GTPs
      picked is randomized (see below).
    - Issues SPARQL queries to find possible fixations for the selected variable
      under the previously selected GTPs subset. Counts the fixation's
      occurrences wrt. the GTPs and sorts the result descending by these counts.
    - Limits the result rows to deal with potential long-tails.
    - Filters the resulting rows with mutate_fix_var_filter.
    - From the limited, filtered result rows randomly selects up to sample_max_n
      candidate fixations with probabilities according to their counts.
    - For each candidate fixation returns a child in which rand_var is replaced
      with the candidate fixation.

    The reasons for fixing rand_var based on a randomly sized subset of GTPs
    are efficiency and shadowing problems with common long-tails. Due to the
    later imposed limit (which is vital in real world use-cases),
    a few remaining GTPs that share more than `limit` potential fixations (so
    have a common long-tail) could otherwise hide solutions for other
    remaining GTPs. This can be the case if these common fixations have low
    fitness. By randomizing the subset size, we will eventually (and more
    likely) select other combinations of remaining GTPs.

    :param sparql: SPARQLWrapper endpoint.
    :param timeout: Timeout in seconds for each individual query (gp).
    :param gtp_scores: Current GTPScores object for sampling.
    :param child: a graph pattern to mutate.
    :param gtp_sample_max_n: Maximum GTPs subset size to base fixations on.
    :param rand_var: If given uses this variable instead of a random one.
    :param sample_max_n: Maximum number of children.
    :param limit: SPARQL limit for the top-k result rows.
    :return: A list of children in which the selected variable is substituted
        with fixation candidates wrt. GTPs.
    """
    assert isinstance(child, GraphPattern)
    assert isinstance(gtp_scores, GTPScores)

    # The further we get, the less gtps are remaining. Sampling too many (all)
    # of them might hurt as common substitutions (> limit ones) which are dead
    # ends could cover less common ones that could actually help
    gtp_sample_max_n = min(gtp_sample_max_n, int(gtp_scores.remaining_gain))
    gtp_sample_max_n = random.randint(1, gtp_sample_max_n)

    ground_truth_pairs = gtp_scores.remaining_gain_sample_gtps(
        max_n=gtp_sample_max_n)
    rand_vars = child.vars_in_graph - {SOURCE_VAR, TARGET_VAR}
    if len(rand_vars) < 1:
        return [child]
    if rand_var is None:
        rand_var = random.choice(list(rand_vars))
    t, substitution_counts = variable_substitution_query(
        sparql, timeout, child, rand_var, ground_truth_pairs, limit)
    if not substitution_counts:
        # the current pattern is unfit, as we can't find anything fulfilling it
        logger.debug("tried to fix a var %s without result:\n%s"
                     "seems as if the pattern can't be fulfilled!",
                     rand_var, child)
        return [child]

    mutate_fix_var_filter(substitution_counts)
    if not substitution_counts:
        # could have happened that we removed the only possible substitution
        return [child]
    # randomly pick n of the substitutions with a prob ~ to their counts
    items, counts = zip(*substitution_counts.most_common())
    substs = sample_from_list(items, counts, sample_max_n)
    logger.log(
        config.LOGLVL_MUTFV,
        'fixed variable %s in %sto:\n %s\n<%d out of:\n%s\n',
        rand_var.n3(),
        child,
        '\n '.join([subst.n3() for subst in substs]),
        sample_max_n,
        '\n'.join([' %d: %s' % (c, v.n3())
                   for v, c in substitution_counts.most_common()]),
    )
    res = [
        GraphPattern(child, mapping={rand_var: subst})
        for subst in substs
    ]
    return res


def mutate_deep_narrow_path(
        sparql,
        timeout,
        gtp_scores,
        child,
        directions=None,
        child_in_queries=False,
        limit=None,  # TODO: Use a limit for the queries?
):
    """ Finds n-hop-connections from Source to Target, to add them to a given
    Graph-Pattern.
    
    The outline of the mutation is as follows:
    - If not evaluated, evaluates the given GP to work on its matching-node-
      pairs
    - If not passed in, randomly selects the path-length and the directions
      of the single hops.
    - Issues SPARQL queries, to find hops (from Source and Target), that don't
      have a big fan-out (smaller than the default-value). Uses an default max-
      amount of found hops to find the next hop.
      When there is only one hop left to find, it tries to instanciate paths,
      that fit to an STP. If such a path is found, its hops are added to the GP.
      As there could be more than one path, the mutation returns a list of such
      patterns.

    :param directions: list of directions to use for the hops
        (1: Source -> Target, -1: Target -> Source,
        0 (or everything else): choose random)
    :param child_in_queries: If true: add the triples of the given pattern to
        the queries
    :param limit: SPARQL limnit
    :return: list of children in which a deep_narrow_path is added
    """
    if not child.fitness.valid:
        ev = evaluate(
            sparql, timeout, gtp_scores, child, run=-1, gen=-1)
        update_individuals([child], [ev])
    gtps = child.matching_node_pairs
    if not gtps:
        return [child]
    if directions:
        n = len(directions) - 1
    else:
        alpha = config.MUTPB_DN_MAX_HOPS_ALPHA
        beta = config.MUTPB_DN_MAX_HOPS_BETA
        max_hops = config.MUTPB_DN_MAX_HOPS
        # more likely to create shorter paths
        # with default values the distribution is as follows:
        # PDF: 1: 14 %, 2: 27 %, 3: 25 %, 4: 17 %, 5: 10 %, 6: 5 %, 7: 1.5 %, ...
        # CDF: 1: 14 %, 2: 40 %, 3: 66 %, 4: 83 %, 5: 93 %, 6: 98 %, 7: 99,6 %, ...
        n = int(random.betavariate(alpha, beta) * max_hops + 1)
    nodes = [SOURCE_VAR] + [Variable('n%d' % i) for i in range(n)] + [TARGET_VAR]
    hops = [Variable('p%d' % i) for i in range(n + 1)]
    if not directions:
        directions = [0 for _ in range(n + 1)]
    directions = [
        random.choice([-1, 1]) if d not in [-1, 1] else d for d in directions
    ]
    gp_hops = [
        # directions[i] == 1 => hop in the direction source -> target
        GraphPattern([(nodes[i], hops[i], nodes[i + 1])]) if directions[i] == 1
        # directions[i] == -1 => hop in the direction target -> source
        else GraphPattern([(nodes[i + 1], hops[i], nodes[i])])
        for i in range(n+1)
    ]
    # queries to get the first n hops:
    valueblocks_s = {}
    valueblocks_t = {}
    for i in range(n // 2 + 1):
        if i < int(n/2):
            t, q_res = deep_narrow_path_query(
                sparql,
                timeout,
                child,
                hops[i],
                nodes[i+1],
                valueblocks_s,
                gp_hops[:i + 1],
                SOURCE_VAR,
                gp_in=child_in_queries,
            )
            if not q_res:
                return [child]
            valueblocks_s[hops[i]] = {
                (hops[i],): random.sample(
                    [(q_r,) for q_r in q_res],
                    min(config.MUTPB_DN_MAX_HOP_INST, len(q_res))
                )
            }
        if n-i > i:
            t, q_res = deep_narrow_path_query(
                sparql,
                timeout,
                child,
                hops[n-i],
                nodes[n-i],
                valueblocks_t,
                gp_hops[n - i:],
                TARGET_VAR,
                gp_in=child_in_queries,
            )
            if not q_res:
                return [child]
            valueblocks_t[hops[n-i]] = {
                (hops[n-i],): random.sample(
                    [(q_r,) for q_r in q_res],
                    min(config.MUTPB_DN_MAX_HOP_INST, len(q_res))
                )
            }

    # query to get the last hop and instantiations, that connect source and
    # target
    valueblocks = {}
    valueblocks.update(valueblocks_s)
    valueblocks.update(valueblocks_t)
    t, q_res = deep_narrow_path_inst_query(
        sparql,
        timeout,
        child,
        hops,
        valueblocks,
        gp_hops,
        gp_in=child_in_queries
    )
    if not q_res:
        return [child]
    res = [
        child + GraphPattern([
            (nodes[i], qr[i], nodes[i + 1]) if directions[i] == 1
            else (nodes[i + 1], qr[i], nodes[i])
            for i in range(n + 1)
        ]) for qr in q_res
    ]
    return res


def mutate_simplify_pattern(gp):
    if len(gp) < 2:
        return gp
    orig_gp = gp
    logger.debug('simplifying pattern\n%s', gp)

    # remove parallel variable edges (single variables only)
    # e.g., [ :x ?v1 ?y . :x ?v2 ?y. ] should remove :x ?v2 ?y.
    identifier_counts = gp.identifier_counts()
    edge_vars = [edge for edge in gp.edges if isinstance(edge, Variable)]
    # note that we also count occurrences in non-edge positions just to be safe!
    edge_var_counts = Counter({v: identifier_counts[v] for v in edge_vars})
    edge_vars_once = [var for var, c in edge_var_counts.items() if c == 1]
    for var in sorted(edge_vars_once, reverse=True):
        var_triple = [(s, p, o) for s, p, o in gp if p == var][0]  # only one
        s, _, o = var_triple
        parallel_triples = [
            t for t in gp if (t[0], t[2]) == (s, o) and t[1] != var
        ]
        if parallel_triples:
            # remove alpha-num largest var triple
            gp -= [var_triple]

    # remove edges between fixed nodes (fixed and single var edges from above)
    fixed_node_triples = [
        (s, p, o) for s, p, o in gp
        if not isinstance(s, Variable) and not isinstance(o, Variable)
    ]
    gp -= [
        (s, p, o) for s, p, o in fixed_node_triples
        if not isinstance(p, Variable) or p in edge_vars_once
    ]

    # remove unrestricting leaf edges (single occurring vars only) and leaves
    # behind fixed nodes
    # more explicit: such edges are
    # - single occurrence edge vars with a single occ gen var node,
    #   so (x, ?vp, ?vn) or (?vn, ?vp, x)
    # - or single occ gen var node behind fixed nodes
    old_gp = None
    while old_gp != gp:
        old_gp = gp
        gen_var_counts = gp.var_counts()
        for i in (SOURCE_VAR, TARGET_VAR):
            # remove all non generated vars from gen_var_counts
            del gen_var_counts[i]
        for t in gp:
            counts = map(lambda x: gen_var_counts[x], t)
            s, p, o = t
            # get counts for s, p and p, o. if [1, 1] remove triple
            if ((
                # single occ edge var with single node var
                counts[0:2] == [1, 1] or counts[1:3] == [1, 1]
            ) or (
                # single occ gen var node behind fixed node
                counts[1] == 0 and (
                    (not isinstance(s, Variable) and counts[2] == 1) or
                    (not isinstance(o, Variable) and counts[0] == 1)
                )
            )):
                gp -= [t]

    # TODO: parallel edges like
    # ?s <p> ?v1 . ?v1 <q> ?t .
    # ?s <p> ?v2 . ?v2 <q> ?t .

    # TODO: remove fixed edge only connected patterns like
    # ?s <p> ?t . <x> <p> ?v1 . ?v1 ?v2 <y> .

    # TODO: maybe remove disconnected components (relevance in reality?)

    if len(gp) < 1:
        # for example: ?s ?v1 ?v2 .
        logger.log(
            config.LOGLVL_MUTSP,
            'simplification of the following pattern resulted in empty pattern,'
            ' returning original pattern:\n%s',
            orig_gp,
        )
        return orig_gp

    if orig_gp == gp:
        logger.log(
            config.LOGLVL_MUTSP,
            'simplification had no effect on pattern:\n%s',
            gp,
        )
    else:
        logger.log(
            config.LOGLVL_MUTSP,
            'successfully simplified pattern:\n%swas simplified to:\n%s',
            orig_gp,
            gp,
        )
    return gp


@exception_stack_catcher
def mutate(
        sparql,
        timeout,
        gtp_scores,
        child,
        pb_ae=config.MUTPB_AE,
        pb_dt=config.MUTPB_DT,
        pb_en=config.MUTPB_EN,
        pb_fv=config.MUTPB_FV,
        pb_id=config.MUTPB_ID,
        pb_iv=config.MUTPB_IV,
        pb_mv=config.MUTPB_MV,
        pb_sp=config.MUTPB_SP,
        pb_sv=config.MUTPB_SV,
        pb_dn=config.MUTPB_DN,
):
    # mutate patterns:
    # grow: select random identifier and convert them into a var (local)
    # grow: select var and randomly split in 2 vars (local)
    # shrink: merge 2 vars (local)
    # shrink: del triple (local)
    # grow: select random node, add edge with / without neighbor (local for now)
    # grow: select random 2 nodes, add edge between (local for now)
    # grow: increase distance between source and target by moving one a hop away
    # shrink: fix variable (SPARQL)
    # shrink: simplify pattern
    orig_child = child
    assert fit_to_live(child), 'mutation on unfit child: %r' % (child,)

    if random.random() < pb_iv:
        child = mutate_introduce_var(child)
    if random.random() < pb_sv:
        child = mutate_split_var(child)
    if random.random() < pb_mv:
        child = mutate_merge_var(child)
    if random.random() < pb_dt:
        child = mutate_del_triple(child)

    if random.random() < pb_en:
        child = mutate_expand_node(child)
    if random.random() < pb_ae:
        child = mutate_add_edge(child)

    if random.random() < pb_id:
        child = mutate_increase_dist(child)

    if random.random() < pb_sp:
        child = mutate_simplify_pattern(child)

    if random.random() < pb_fv:
        child = canonicalize(child)
        children = mutate_fix_var(sparql, timeout, gtp_scores, child)
    else:
        if random.random() < pb_dn:
            children = mutate_deep_narrow_path(sparql, timeout, gtp_scores, child)
        else:
            children = [child]

    children = {
        c if fit_to_live(c) else orig_child
        for c in children
    }
    children = {
        canonicalize(c) for c in children
    }
    return list(children)


def train(toolbox, population, run):
    hall_of_fame = deap.tools.HallOfFame(config.HOFSIZE)
    # pop = toolbox.population(n=50)
    pop = population
    g = 0
    logger.info(
        'Run %d, Generation %d: %d individuals',
        run, g, len(pop)
    )
    logger.debug('Population: %r', pop)

    # Evaluate the entire population
    _evaluate = partial(toolbox.evaluate, run=run, gen=g)
    eval_results = list(parallel_map(_evaluate, pop))
    logger.info(
        'Run %d, Generation %d: evaluated %d individuals',
        run, g, len(pop)
    )
    logger.debug('Evaluation results: %r', eval_results)
    update_individuals(pop, eval_results)
    hall_of_fame.update(pop)
    best_individual = hall_of_fame[0]
    best_individual_gen = g
    if not toolbox.generation_step_callback(g, pop):
        logger.info(
            'terminating learning as requested by generation_step_callback'
        )
        return g, pop, hall_of_fame

    # TODO: don't double eval same pattern? maybe a bit redundancy is good?
    # TODO: increase timeout if > x % of population fitnesses show timeout

    for g in range(1, config.NGEN + 1):
        # Select the next generation individuals
        offspring = toolbox.select(pop)
        logger.info(
            "Run %d, Generation %d: selected %d offspring individuals",
            run, g, len(offspring)
        )
        logger.debug('Offspring: %r', offspring)
        # Clone the selected individuals
        # offspring = map(toolbox.clone, offspring)

        # Apply crossover and mutation on the offspring
        tmp = []
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < config.CXPB:
                child1, child2 = toolbox.mate(child1, child2)
            tmp.append(child1)
            tmp.append(child2)
        offspring = tmp
        logger.info(
            "Run %d, Generation %d: %d individuals after mating",
            run, g, len(offspring)
        )
        logger.debug('Offspring: %r', offspring)

        mutants = []
        tmp = []
        for child in offspring:
            if random.random() < config.MUTPB:
                mutants.append(child)
            else:
                tmp.append(child)
        offspring = tmp
        logger.debug('Offspring in gen %d to mutate: %r', g, mutants)
        mutant_children = list(parallel_map(toolbox.mutate, mutants))
        logger.debug('Mutation results in gen %d: %r', g, mutant_children)
        for mcs in mutant_children:
            offspring.extend(mcs)
        logger.info(
            "Run %d, Generation %d: %d individuals after mutation",
            run, g, len(offspring)
        )
        logger.debug('Offspring: %r', offspring)

        # don't completely replace pop, but keep good individuals
        # will draw individuals from the first 10 % of the HOF
        # CDF: 0: 40 %, 1: 64 %, 2: 78 %, 3: 87 %, 4: 92 %, 5: 95 %, ...
        l = len(hall_of_fame)
        offspring += [
            hall_of_fame[int(random.betavariate(1, 50) * (l - 1))]
            for _ in range(config.HOFPAT_REINTRO)
        ]
        # always re-introduce some variable patterns with ?source and ?target
        offspring += [
            child for child in generate_variable_patterns(config.VARPAT_REINTRO)
            if fit_to_live(child)
        ]
        logger.info(
            'Run %d, Generation %d: %d individuals after re-adding hall-of-fame'
            ' and variable patterns',
            run, g, len(offspring)
        )
        logger.debug('Offspring: %r', offspring)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        logger.info(
            "Run %d, Generation %d: %d individuals to evaluate",
            run, g, len(invalid_ind)
        )
        logger.debug('Evaluating individuals in gen %d: %r', g, invalid_ind)
        _evaluate = partial(toolbox.evaluate, run=run, gen=g)
        eval_results = list(parallel_map(_evaluate, invalid_ind))
        logger.info(
            "Run %d, Generation %d: %d individuals evaluated",
            run, g, len(eval_results)
        )
        logger.debug('Evaluation results in gen %d: %r', g, eval_results)
        update_individuals(invalid_ind, eval_results)
        hall_of_fame.update(invalid_ind)

        # replace population with this generation's offspring
        pop[:] = offspring
        logger.info(
            "Run %d, Generation %d: %d individuals",
            run, g, len(pop)
        )
        logger.debug('Population of generation %d: %r', g, pop)

        if not toolbox.generation_step_callback(g, pop):
            logger.info(
                'terminating learning as requested by generation_step_callback'
            )
            break

        if best_individual.fitness < hall_of_fame[0].fitness:
            best_individual = hall_of_fame[0]
            best_individual_gen = g

        if g >= best_individual_gen + config.NGEN_NO_IMPROVEMENT:
            logger.info(
                'terminating learning after generation %d: '
                'no better individual found since best in generation %d.',
                g, best_individual_gen
            )
            break

    return g, pop, hall_of_fame


def generate_variable_pattern(dist):
    # generates a variable pattern with a given dist between source and target.
    # TODO: optimize by directly picking from all possible patterns?
    # the idea here is that we need some kind of "walk" / connection between
    # source and target and all else relies on mutations and mating to form more
    # complicated patterns. It's possible though, that actually just starting
    # from random triple patterns would speed up the algorithm, so maybe make
    # it an alternative?
    assert dist > 0
    # path of properties and targets of length dist ending at TARGET_VAR
    ps = [gen_random_var() for _ in range(dist)]
    ts = [gen_random_var() for _ in range(dist - 1)] + [TARGET_VAR]
    s = SOURCE_VAR  # start at source
    triples = []
    for p, t in zip(ps, ts):
        e = (s, p, t)
        if random.random() < .5:  # edge in any direction
            e = (t, p, s)
        triples.append(e)
        s = t
    return canonicalize(GraphPattern(triples))


def generate_variable_patterns(count):
    u = Variable('u')
    p = Variable('p')
    v = Variable('v')

    s_out = canonicalize(GraphPattern([(SOURCE_VAR, p, v)]))
    s_in = canonicalize(GraphPattern([(u, p, SOURCE_VAR)]))
    t_out = canonicalize(GraphPattern([(TARGET_VAR, p, v)]))
    t_in = canonicalize(GraphPattern([(u, p, TARGET_VAR)]))
    res = []

    alpha = config.INIT_POP_LEN_ALPHA
    beta = config.INIT_POP_LEN_BETA
    for _ in range(count):
        # more likely to create shorter variable patterns, only few incomplete
        # with default values the distribution is as follows:
        # PDF: 0: 7 %, 1: 41 %, 2: 35 %, 3: 13 %, 4: 2.7 %, 5: 0.4 %, ...
        # CDF: 0: 7 %, 1: 48 %, 2: 84 %, 3: 96.8 %, 4: 99.5 %, 5: 99.9 %, ...
        dist = int(random.betavariate(alpha, beta) * config.MAX_PATTERN_LENGTH)
        if dist < 1:
            gp = random.choice((s_out, s_in, t_out, t_in),)
        else:
            gp = generate_variable_pattern(dist)
        res.append(gp)
    # can happen that pattern is too long and thereby not fit_to_live:
    # can be > config.MAX_PATTERN_LENGTH or > config.MAX_PATTERN_VARS.
    # The latter is sometimes desired if those variables go through
    # mutate_fix_var before ending up in any population
    return res


def generate_init_population(
        sparql, timeout, gtp_scores,
        pb_fv=config.INIT_POPPB_FV,
        n=config.INIT_POPPB_FV_N,
        init_patterns=None,
        pb_init_pattern=config.INIT_POPPB_INIT_PAT,
):
    logger.info('generating init population of seed size %d', config.POPSIZE)
    population = []

    # Variable patterns:
    var_pats = generate_variable_patterns(config.POPSIZE)

    if init_patterns:
        # replace var pats with random ones from init_patterns according to prob
        var_pats = [
            p for p in var_pats
            if random.random() > pb_init_pattern
        ]
        for _ in range(len(var_pats), config.POPSIZE):
            var_pats.append(random.choice(init_patterns).copy())
        random.shuffle(var_pats)

    # initial run of mutate_fix_var to instantiate many of the variable patterns
    # TODO: maybe loop this? (why only try to fix one var?)
    to_fix = []
    for vp in var_pats:
        if random.random() < pb_fv:
            to_fix.append(vp)
        else:
            population.append(vp)
    logger.info(
        'initial mutate_fix_var run on %d individuals', len(to_fix)
    )
    p_mfv = partial(mutate_fix_var, sparql, timeout, gtp_scores, sample_max_n=n)
    fixed_result_patterns_per_vp = list(parallel_map(p_mfv, to_fix))
    for fixed_result_patterns in fixed_result_patterns_per_vp:
        population.extend(fixed_result_patterns)

    # rare possibility that patterns with too many vars are generated
    fit_population = [gp for gp in population if fit_to_live(gp)]

    l = len(fit_population)
    logger.info(
        'after initial mutate_fix_var run init population now has %d '
        'individuals', l
    )
    if l < config.POPSIZE:
        lvl = logging.DEBUG
        if l < 0.95 * config.POPSIZE:
            lvl = logging.INFO
        if l < 0.50 * config.POPSIZE:
            lvl = logging.WARNING
        sample_unfit = random.sample(
            [gp for gp in population if gp not in fit_population],
            min(10, len(population) - l)
        )
        logger.log(
            lvl,
            'generated init population of size %d < POPSIZE = %d\n'
            'check these config variables:\n'
            '  MAX_PATTERN_LENGTH: %d\n'
            '  MAX_PATTERN_VARS: %d\n'
            '  INIT_POP_LEN_ALPHA: %.3f\n'
            '  INIT_POP_LEN_BETA: %.3f\n'
            '  INIT_POPPB_FV: %.3f\n'
            '  INIT_POPPB_FV_N: %.3f\n'
            'it seems they are selected in a way that we are generating not '
            'fit_to_live patterns in generate_variable_patterns().\n'
            'var_pats: %d\n'
            '%d graph patterns before dropping %d like these (samples):\n%s',
            l, config.POPSIZE,
            config.MAX_PATTERN_LENGTH,
            config.MAX_PATTERN_VARS,
            config.INIT_POP_LEN_ALPHA,
            config.INIT_POP_LEN_BETA,
            config.INIT_POPPB_FV,
            config.INIT_POPPB_FV_N,
            len(var_pats),
            len(population),
            len(population) - l,
            ''.join(str(gp) for gp in sample_unfit)
        )

    return fit_population


def generation_step_callback(
        run, gtp_scores, user_callback_per_generation, ngen, population
):
    """Called after each generation step cycle in train().

    :param run: number of the current run
    :param gtp_scores: gtp_scores as of start of this run
    :param user_callback_per_generation: a user provided callback that is called
        after each training generation. If not None called like this:
        user_callback_per_generation(run, gtp_scores, ngen, population)
        If user_callback_per_generation returns anything else than None, it
        overrides our return condition.
    :param ngen: the number of the current generation.
    :param population: the current population after generation ngen.
    :return: If user_callback_per_generation returns anything but None, it is
        used as the return value. Else, if config.QUICK_STOP and the current
        population is good enough to cover the remaining gains so we end up
        below the MIN_REMAINING_GAIN, we signal train to quick-stop by returning
        False. Otherwise (the likely default), we signal train to continue
        training by returning True.
    """
    assert isinstance(gtp_scores, GTPScores)
    top_counter = print_population(run, ngen, population)
    top_gps = sorted(
        top_counter.keys(), key=attrgetter("fitness"), reverse=True
    )
    generation_gtp_scores = gtp_scores.copy_reset()
    generation_gtp_scores.update_with_gps(top_gps)
    save_generation(
        run, ngen, top_gps, generation_gtp_scores
    )

    qsbs = [v for v in parallel_map(query_stats, [(run, ngen)] * 1000) if v]
    qs, bs = zip(*qsbs)
    qs = sum(qs)
    bs = Counter(bs).most_common()
    logger.info('QueryStats totals:\n  Batch-Sizes: %s\n%s', kv_str(bs), qs)

    pause_if_signaled_by_file()
    if user_callback_per_generation:
        # user provided callback
        res = user_callback_per_generation(run, gtp_scores, ngen, population)
        if res is not None:
            return res

    pre_gtp_scores = gtp_scores.copy()
    pre_gtp_scores.update_with_gps(population)
    rem_gain = gtp_scores.remaining_gain
    pot_rem_gain = pre_gtp_scores.remaining_gain
    pot_gain = rem_gain - pot_rem_gain
    l = len(gtp_scores)
    logger.info(
        "Run %d:\n"
        "  remains: %.1f / %d total = %.1f %%\n"
        "  coverage: %.1f / %d total = %.1f %%\n"
        "Generation %d:\n"
        "  potential remains: %.1f (%.1f %%)\n"
        "  potential coverage: %.1f (%.1f %%)\n"
        "Potential gain: %.1f (%.1f %%)\n"
        '(without run post-processing, e.g., min-fitness filtering)',
        run,
        rem_gain, l, rem_gain * 100 / l,
        l - rem_gain, l, (l - rem_gain) * 100 / l,
        ngen,
        pot_rem_gain, pot_rem_gain * 100 / l,
        l - pot_rem_gain, (l - pot_rem_gain) * 100 / l,
        pot_gain, pot_gain * 100 / l,
    )
    return not check_quick_stop(pre_gtp_scores)


def check_quick_stop(
        pre_gtp_scores,
        quick_stop=config.QUICK_STOP,
        min_remaining_gain=config.MIN_REMAINING_GAIN,
):
    if quick_stop:
        if pre_gtp_scores.remaining_gain < min_remaining_gain:
            logger.info('quick-stop condition reached')
            return True
    return False


def find_graph_patterns(
        sparql, run, gtp_scores,
        init_patterns=None,
        user_callback_per_generation=None,
):
    timeout = calibrate_query_timeout(sparql)

    toolbox = deap.base.Toolbox()

    toolbox.register(
        "mate", mate
    )
    toolbox.register(
        "mutate", mutate, sparql, timeout, gtp_scores,
    )
    toolbox.register(
        "select", deap.tools.selTournament,
        k=config.POPSIZE,
        tournsize=config.TOURNAMENT_SIZE,
    )
    toolbox.register(
        "evaluate", evaluate, sparql, timeout, gtp_scores)
    toolbox.register(
        "generation_step_callback",
        generation_step_callback, run, gtp_scores, user_callback_per_generation
    )


    population = generate_init_population(
        sparql, timeout, gtp_scores,
        init_patterns=init_patterns,
    )

    # noinspection PyTypeChecker
    ngen, res_population, hall_of_fame = train(toolbox, population, run)

    print("\n\n\nhall of fame:")
    for r in hall_of_fame[:20]:
        assert isinstance(r, GraphPattern)
        print_graph_pattern(r)

    return ngen, res_population, hall_of_fame, toolbox


def calc_min_fitness(gtp_scores, min_score):
    """Calculates the minimum desired fitness in the current run.

    In each run, the fitness tuple has a first component "remains", which is
    constant. The score is what we actually want to set to min_score.
    """
    min_fitness = GPFitnessTuple(
        remains=gtp_scores.remaining_gain,
        score=min_score,
    )
    return GPFitness(min_fitness)


def _find_graph_pattern_coverage_run(
        sparql,
        min_score,
        run,
        coverage_counts,
        gtp_scores,
        patterns,
        init_patterns=None,
        user_callback_per_generation=None,
        user_callback_per_run=None,
):
    min_fitness = calc_min_fitness(gtp_scores, min_score)

    ngen, res_pop, hall_of_fame, toolbox = find_graph_patterns(
        sparql, run, gtp_scores,
        init_patterns=init_patterns,
        user_callback_per_generation=user_callback_per_generation,
    )

    # TODO: coverage patterns should be chosen based on similarity
    new_best_patterns = []
    for pat in hall_of_fame:
        if pat.fitness < min_fitness:
            logging.info(
                'skipping remaining patterns cause they are below min_fitness:'
                '\n%s',
                min_fitness.format_fitness()
            )
            break
        s_pat = canonicalize(mutate_simplify_pattern(pat))
        if pat in patterns:
            logger.info(
                'found pattern again, skipping for graph pattern coverage:\n'
                '%s',
                format_graph_pattern(pat, 0),
            )
        elif s_pat in patterns:
            logger.info(
                'found pattern again (simpler version already in results), '
                'skipping for graph pattern coverage:\n'
                'Orig:\n%sSimplified:\n%s',
                format_graph_pattern(pat, 0),
                format_graph_pattern(s_pat, 0)
            )
        else:
            if pat != s_pat:
                # seems the current pattern isn't as simple as possible,
                # check if the simplified version is better (expected)
                # make sure s_pat has a fitness
                # noinspection PyUnresolvedReferences
                update_individuals([s_pat], [toolbox.evaluate(s_pat)])
                # noinspection PyProtectedMember
                if s_pat.fitness >= pat.fitness:
                    logging.info(
                        'using improved result pattern by simplification:\n'
                        'Orig:\n%sSimplified:\n%s',
                        format_graph_pattern(pat, 0),
                        format_graph_pattern(s_pat, 0),
                    )
                    pat = s_pat
                elif (s_pat.fitness.values._replace(qtime=0) ==
                        pat.fitness.values._replace(qtime=0)):
                    # can happen that just vars were renamed and
                    # the simplified query took a bit longer:
                    logging.info(
                        'using canonical pattern even though a bit slower:\n'
                        'Orig:\n%sSimplified:\n%s',
                        format_graph_pattern(pat, 0),
                        format_graph_pattern(s_pat, 0),
                    )
                    pat = s_pat
                else:
                    if s_pat.fitness.values.timeout:
                        logging.info(
                            'simplified pattern has worse fitness and timed '
                            'out, using original instead:\n'
                            'Orig:\n%sSimplified:\n%s',
                            format_graph_pattern(pat, 0),
                            format_graph_pattern(s_pat, 0),
                        )
                    else:
                        logger.warning(
                            'simplified pattern has worse fitness, using '
                            'original instead:\n'
                            'Orig:\n%sSimplified:\n%s',
                            format_graph_pattern(pat, 0),
                            format_graph_pattern(s_pat, 0),
                        )
            logger.info(
                'found new pattern for graph pattern coverage:\n%s',
                format_graph_pattern(pat, 1000),
            )
            new_best_patterns.append((pat, run))

    # finally update gtp_scores with the new patterns (don't do this before as
    # evaluate of simplified patterns would otherwise have different remains and
    # thereby always return inferior results)
    for pat, run in new_best_patterns:
        coverage_counts.update(pat.matching_node_pairs)
    new_best_gps = [gp for gp, _ in new_best_patterns]
    gtp_scores.update_with_gps(new_best_gps)

    run_gtp_scores = gtp_scores.copy_reset()
    run_gtp_scores.update_with_gps(new_best_gps)
    save_run(
        new_best_patterns,
        coverage_counts=coverage_counts,
        run_gtp_scores=run_gtp_scores,
        overall_gtp_scores=gtp_scores,
        run=run,
    )

    if user_callback_per_run:
        user_callback_per_run(
            run, gtp_scores, new_best_patterns, coverage_counts
        )

    return new_best_patterns, coverage_counts, gtp_scores


def find_graph_pattern_coverage(
        sparql,
        ground_truth_pairs,
        init_patterns=None,
        min_score=config.MIN_SCORE,
        min_remaining_gain=config.MIN_REMAINING_GAIN,
        max_runs=config.NRUNS,
        runs_no_improvement=config.NRUNS_NO_IMPROVEMENT,
        error_retries=config.ERROR_RETRIES,
        user_callback_per_generation=None,
        user_callback_per_run=None,
):
    assert isinstance(ground_truth_pairs, tuple)

    logger.info(
        'Started learning:\n'
        'NRUNS=%d, NRUNS_NO_IMPROVEMENT=%d,\n'
        'NGEN=%d, NGEN_NO_IMPROVEMENT=%d',
        config.NRUNS, config.NRUNS_NO_IMPROVEMENT,
        config.NGEN, config.NGEN_NO_IMPROVEMENT,
    )

    error_count = 0
    # the following are modified in-place by _find_graph_pattern_coverage_run()
    ground_truth_pairs = list(ground_truth_pairs)
    coverage_counts = Counter({gtp: 0 for gtp in ground_truth_pairs})
    gtp_scores = GTPScores(ground_truth_pairs)
    patterns = {}
    last_pattern_update_in_run = 0
    run = 1
    while run <= max_runs:
        # noinspection PyBroadException
        try:
            # run in a loop and remove ground-truth pairs that
            # are best matched until we have good patterns for all gt pairs
            remaining_gain = gtp_scores.remaining_gain
            if remaining_gain < min_remaining_gain:
                break

            prev_run = find_run_result(run)
            if prev_run:
                res = load_results(prev_run)
            else:
                res = _find_graph_pattern_coverage_run(
                    sparql,
                    min_score,
                    run,
                    coverage_counts,
                    gtp_scores,
                    patterns,
                    init_patterns=init_patterns,
                    user_callback_per_generation=user_callback_per_generation,
                    user_callback_per_run=user_callback_per_run,
                )
            new_best_patterns, coverage_counts, gtp_scores = res
            patterns.update({pat: run for pat, run in new_best_patterns})
            new_remaining_gain = gtp_scores.remaining_gain
            if new_remaining_gain < remaining_gain:
                last_pattern_update_in_run = run
                logger.info(
                    'coverage improvement in run %d: %.2f - %.2f = %.2f',
                    run, remaining_gain, new_remaining_gain,
                    remaining_gain - new_remaining_gain
                )
            else:
                logger.info('no coverage improvement in run %d', run)
                if run >= last_pattern_update_in_run + runs_no_improvement:
                    logger.info(
                        'no coverage improvement in the last %d runs, stopping',
                        runs_no_improvement
                    )
                    break
            run += 1
        except GPLearnerAbortException as e:
            # this exception is only raised to intentionally abort the whole
            # learning process. Don't run auto-retry.
            logger.info('gp learner was aborted intentionally: %r', e)
            raise
        except Exception as e:
            error_count += 1
            logger.error('uncaught exception in run %d', run)
            log_wrapped_exception(logger, e)
            if error_count > error_retries:
                logger.error(
                    'giving up after %d > %d errors',
                    error_count, error_retries
                )
                raise
            else:
                logger.error(
                    'this was uncaught exception number %d, will retry in %ds '
                    'despite error...',
                    error_count, config.ERROR_WAIT
                )
                logging_config.save_error_logs()
                sleep(config.ERROR_WAIT)

    # sort patterns by fitness, run and then pattern
    patterns = sorted(
        patterns.items(),
        key=lambda x: ([-v for v in x[0].fitness.wvalues], x[1], x[0])
    )
    return patterns, coverage_counts, gtp_scores


def predict_target_candidates(
        sparql, timeout, gps, source, parallel=None, exclude_source=None):
    """Uses the gps to predict target candidates for the given source.

    :param sparql: SPARQLWrapper endpoint.
    :param timeout: Timeout in seconds for each individual query (gp).
    :param gps: A list of evaluated GraphPattern objects (fitness is used).
    :param source: source node for which to predict target candidates.
    :param parallel: execute prediction queries in parallel?
    :param exclude_source: remove targets that are source node?
    :return: A list of target_candidate sets for each gp.
    """
    if parallel is None:
        parallel = config.PREDICTION_IN_PARALLEL
    if exclude_source is None:
        exclude_source = config.PREDICTION_EXCLUDE_SOURCE

    pq = partial(
        predict_query,
        sparql, timeout,
        source=source,
    )
    map_ = parallel_map if parallel else map
    results = map_(pq, gps)
    # drop timings:
    res = [target_candidates for _, target_candidates in results]
    if exclude_source:
        res = [tcs - {source} for tcs in res]
    return res


def predict_multi_target_candidates(
        sparql, timeout, gps, sources, parallel=None, exclude_source=None):
    """Uses the gps to predict target candidates for the given source.

    :param sparql: SPARQLWrapper endpoint.
    :param timeout: Timeout in seconds for each individual query (gp).
    :param gps: A list of evaluated GraphPattern objects (fitness is used).
    :param sources: source nodes for which to predict target candidates.
    :param parallel: execute prediction queries in parallel?
    :param exclude_source: remove targets that are source node?
    :return: A list containing a dict of source to set(tcs) for each gp.
    """
    assert len(sources) > 1 and isinstance(sources[0], (URIRef, Literal))
    if parallel is None:
        parallel = config.PREDICTION_IN_PARALLEL
    if exclude_source is None:
        exclude_source = config.PREDICTION_EXCLUDE_SOURCE

    pq = partial(
        predict_multi_query,
        sparql, timeout,
        sources=sources,
    )
    map_ = parallel_map if parallel else map
    results = map_(pq, gps)
    # drop timings:
    res = [stcs for _, stcs in results]
    if exclude_source:
        res = [
            OrderedDict([
                (s, tcs - {s})
                for s, tcs in stcs.items()
            ])
            for stcs in res
        ]
    return res


def predict_fused_targets(
        sparql, timeout, gps, source,
        parallel=None, fusion_methods=None, exclude_source=None,
):
    """Predict candidates and fuse the results."""
    return fuse_prediction_results(
        gps,
        predict_target_candidates(
            sparql, timeout, gps, source, parallel, exclude_source),
        fusion_methods
    )


def find_in_prediction(prediction, target):
    try:
        targets, scores = zip(*prediction)
        return targets.index(target)
    except ValueError:
        return -1


def format_prediction_results(method, res, target=None, idx=None, n=10):
    assert not ((target is None) ^ (idx is None)), \
        "target and idx should both be None or neither"
    rs = [
        '  Top %d predictions (method: %s)%s' % (
            n, method,
            (", target at idx: %d" % idx) if idx is not None else ''
        )
    ]
    for t, score in res[:n]:
        rs.append(
            '  ' + ('->' if t == target else '  ') +
            '%s (%.3f)' % (t.n3(), score)
        )
    return '\n'.join(rs)


def print_prediction_results(method, res, target=None, idx=None, n=10):
    print(format_prediction_results(method, res, target, idx, n))


def evaluate_predictions(
        sparql, gps, gtps,
        gtp_predicted_fused_targets=None, fusion_methods=None):
    recall = 0
    method_idxs = defaultdict(list)
    method_order = []
    res_lens = []
    timeout = calibrate_query_timeout(sparql)
    for i, (source, target) in enumerate(gtps, 1):
        print('%d/%d: predicting target for %s (ground truth: %s):' % (
            i, len(gtps), source.n3(), target.n3()))
        if gtp_predicted_fused_targets:
            method_res = gtp_predicted_fused_targets[i-1]
        else:
            method_res = predict_fused_targets(
                sparql, timeout, gps, source, fusion_methods=fusion_methods)
        once = False
        if not method_order:
            method_order = method_res.keys()
        for method, res in method_res.items():
            idx = find_in_prediction(res, target)
            if not once:
                once = True
                if idx < 0:
                    print('  target not found')
                else:
                    recall += 1
                n = len(res)
                res_lens.append(n)
                print('  result list length: %d' % n)
            method_idxs[method].append(idx)

            print_prediction_results(method, res, target, idx)

    recall /= len(gtps)
    print("Ground Truth Pairs: %s" % gtps)
    print("Result list lenghts: %s" % res_lens)
    print("Recall of test set: %.5f" % recall)
    for method, indices in [(m, method_idxs[m]) for m in method_order]:
        print("\nIndices for method %s:\n'%s': %s" % (method, method, indices))
        avg_idx = np.average([i for i in indices if i >= 0])
        median_idx = np.median([i for i in indices if i >= 0])
        ranks = np.array(indices, dtype='f8') + 1
        # noinspection PyTypeChecker
        mrr = np.sum(1 / ranks[ranks > 0]) / len(indices)
        # noinspection PyTypeChecker
        ndcg = np.sum(1 / np.log2(1 + ranks[ranks > 0])) / len(indices)
        # noinspection PyStringFormat
        print(
            "  Avg. index %s: %.3f, Median index: %.3f\n"
            "  MAP (MRR): %.3f, NDCG: %.3f" % (
                method, avg_idx, median_idx, mrr, ndcg))
        recalls_at = [
            (k, len([True for i in indices if k > i >= 0]) / len(indices))
            for k in (1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 75, 100)
        ]
        print("         k:\t%s" % '\t'.join('% 5d' % k for k, r in recalls_at))
        print("  recall@k:\t%s" % '\t'.join('%.3f' % r for k, r in recalls_at))


# noinspection PyUnusedLocal
@log_all_exceptions(logger)
def main(
        sparql_endpoint=config.SPARQL_ENDPOINT,
        associations_filename=None,
        splitting_variant='random',
        train_filename=None,
        test_filename=None,
        swap_source_target=False,
        drop_invalid=False,
        init_patterns_filename=None,
        print_train_test_sets=True,
        reset=False,
        print_topn_raw_patterns=0,
        print_edge_only_connected_patterns=True,
        show_precision_loss_by_query_reduction=False,
        max_queries=0,
        clustering_variant=None,
        print_query_patterns=False,
        predict='',
        fusion_methods=None,
        tests=False,
        **kwds
):
    logging.info('encoding check: äöüß🎅')  # logging utf-8 byte string
    logging.info(u'encoding check: äöüß\U0001F385')  # logging unicode string
    logging.info(u'encoding check: äöüß\U0001F385'.encode('utf-8'))  # convert
    print('encoding check: äöüß🎅')  # printing utf-8 byte string
    print(u'encoding check: äöüß\U0001F385')  # printing unicode string

    # init workers
    init_workers()

    timer_start = datetime.utcnow()
    main_start = timer_start

    gsa = partial(
        get_semantic_associations,
        swap_source_target=swap_source_target,
        drop_invalid=drop_invalid,
    )
    if not train_filename and not test_filename:
        # get semantic association pairs and split in train and test sets
        semantic_associations = gsa(associations_filename)
        assocs_train, assocs_test = split_training_test_set(
            semantic_associations, variant=splitting_variant
        )
    else:
        assocs_train = gsa(train_filename) if train_filename else []
        assocs_test = gsa(test_filename) if test_filename else []
        if predict == 'train_set':
            assert assocs_train, 'trying to train but train file empty'
        if predict == 'test_set':
            assert assocs_test, 'trying to test but test file empty'
    logger.info(
        'training on %d association pairs and testing on %d',
        len(assocs_train), len(assocs_test)
    )
    sys.stdout.flush()
    sys.stderr.flush()

    if print_train_test_sets:
        if assocs_train:
            print(
                "Training Set Source Target Pairs:\n"
                "================================="
            )
            for s, t in assocs_train:
                print("Train: %s %s" % (s.n3(), t.n3()))

        if assocs_test:
            print(
                "\n\n"
                "Test Set Source Target Pairs:\n"
                "============================="
            )
            for s, t in assocs_test:
                print("Test: %s %s" % (s.n3(), t.n3()))

        sys.stdout.flush()
        sys.stderr.flush()


    semantic_associations = tuple(sorted(assocs_train))

    # setup node expander
    sparql = SPARQLWrapper.SPARQLWrapper(sparql_endpoint)

    init_patterns = None
    if init_patterns_filename:
        init_patterns = load_init_patterns(init_patterns_filename)

    if reset:
        remove_old_result_files()
    last_res = find_last_result()
    if not last_res:
        res = find_graph_pattern_coverage(
            sparql, semantic_associations,
            init_patterns=init_patterns,
        )
        result_patterns, coverage_counts, gtp_scores = res
        sys.stderr.flush()

        save_results(
            result_patterns,
            coverage_counts,
            gtp_scores,
        )
        timer_stop = datetime.utcnow()
        logging.info('Training took: %s', timer_stop - timer_start)
    else:
        result_patterns, coverage_counts, gtp_scores = load_results(last_res)
        timer_stop = datetime.utcnow()
        logging.info('Loading model took: %s', timer_stop - timer_start)
    timer_start = timer_stop

    sys.stdout.flush()
    sys.stderr.flush()

    if not result_patterns:
        print("It seems as if no patterns that satisfy your constraints could "
              "be found in training. Consider increasing QUERY_TIMEOUT_MIN, "
              "POPSIZE, decreasing MIN_SCORE or changing other parameters "
              "listed by --help")
        sys.exit(1)

    print_results(
        result_patterns,
        coverage_counts,
        gtp_scores,
        n=print_topn_raw_patterns,
        edge_only_connected_patterns=print_edge_only_connected_patterns,
    )

    gps = [gp for gp, _ in result_patterns]
    print('raw patterns: %d' % len(gps))
    sys.stdout.flush()
    sys.stderr.flush()

    if show_precision_loss_by_query_reduction:
        # amount of requests one wants to make for a prediction
        max_ks = [1, 2, 3, 4, 5, 7] + range(10, 101, 5)
        expected_precision_loss_by_query_reduction(
            gps, semantic_associations, max_ks, gtp_scores,
            variants=[clustering_variant] if clustering_variant else None,
            plot_precision_losses_over_k=True
        )
        sys.stdout.flush()
        sys.stderr.flush()

    # reduce gps by clustering if mandated by max_queries
    gps = cluster_gps_to_reduce_queries(
        gps, max_queries, gtp_scores, clustering_variant)

    if print_query_patterns:
        print(
            '\nusing the following %d graph patterns for prediction:' % len(gps)
        )
        for i, gp in enumerate(gps):
            print('Graph pattern #%d:' % i)
            print_graph_pattern(gp, print_matching_node_pairs=0)

        sys.stdout.flush()
        sys.stderr.flush()

    timer_stop = datetime.utcnow()
    logging.info('Query reduction took: %s', timer_stop - timer_start)
    timer_start = timer_stop

    if predict == 'train_set':
        loaded_predictions = load_predicted_target_candidates()
        gtps = assocs_train if predict == 'train_set' else assocs_test
        if not loaded_predictions:
            print('\n\n\nstarting prediction on %s' % predict)

            timeout = calibrate_query_timeout(sparql)
            gtp_gp_tcs = []
            for i, (source, target) in enumerate(gtps):
                logger.info(
                    '%d/%d: predicting target candidates for source: %s '
                    '(gt target: %s)',
                    i+1, len(gtps), source, target
                )
                gtp_gp_tcs.append(
                    predict_target_candidates(sparql, timeout, gps, source)
                )
            save_predicted_target_candidates(gps, gtps, gtp_gp_tcs)

            sys.stdout.flush()
            sys.stderr.flush()

            timer_stop = datetime.utcnow()
            logging.info('Batch prediction of %s took: %s',
                         predict, timer_stop - timer_start)
            timer_start = timer_stop
        else:
            _gps, _gtps, gtp_gp_tcs = loaded_predictions
            assert gps == _gps, (
                "result patterns learned from previous execution did not match "
                "the current ones (e.g. due to changed --max_queries and/or "
                "clustering). Consider removing generated *.pkl.gz temp files"
            )
            assert gtps == _gtps, (
                "ground truth pairs from previous execution do not match the "
                "current ones (e.g. due to changed --associations_filename). "
                "Consider re-running full training with --reset, running in "
                "manual mode or invoking serve.py."
            )

        train_fusion_models(gps, gtps, gtp_gp_tcs, fusion_methods)
        timer_stop = datetime.utcnow()
        logging.info(
            'Training fusion models took: %s', timer_stop - timer_start)
        timer_start = timer_stop

        logging.info('Batch fusing all prediction candidates...')
        gtp_predicted_fused_targets = [
            fuse_prediction_results(gps, gp_tcs, fusion_methods)
            for gp_tcs in gtp_gp_tcs
        ]
        timer_stop = datetime.utcnow()
        logging.info('Batch fusing all prediction candidates took: %s',
                     timer_stop - timer_start)
        timer_start = timer_stop

        evaluate_predictions(sparql, gps, gtps, gtp_predicted_fused_targets)

    if predict == 'test_set':
        gtps = assocs_test
        print('\n\n\nstarting prediction on %s' % predict)
        evaluate_predictions(sparql, gps, gtps, fusion_methods=fusion_methods)

        sys.stdout.flush()
        sys.stderr.flush()

        timer_stop = datetime.utcnow()
        logging.info('Batch prediction of %s took: %s',
                     predict, timer_stop - timer_start)
        timer_start = timer_stop

    if predict == 'manual':
        timeout = calibrate_query_timeout(sparql)
        sys.stdout.flush()
        sys.stderr.flush()

        while True:
            s = six.moves.input(
                '\n\nEnter a DBpedia resource as source:\n'
                '> http://dbpedia.org/resource/'
            )
            source = URIRef('http://dbpedia.org/resource/' + s)

            method_res = predict_fused_targets(
                sparql, timeout, gps, source, fusion_methods)
            for method, res in method_res.items():
                print_prediction_results(method, res)

    return sparql, gps, fusion_methods
