# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys
from datetime import datetime
from os import path

import SPARQLWrapper
from splendid import make_dirs_for
from splendid import timedelta_to_s

# noinspection PyUnresolvedReferences
import logging_config

# not all import on top due to scoop and init...

GTPS_FILENAME = 'data/dbpedia_random_1000_uri_pairs.csv.gz'
EVAL_DATA_GRAPH = 'urn:gp_learner:eval:data'
RESULT_FILENAME = 'path_length_eval_result.txt'


logger = logging.getLogger(__name__)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description='gp learner path length eval',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--sparql_endpoint',
        help="the SPARQL endpoint",
        action="store",
        default=config.SPARQL_ENDPOINT,
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
        "length",
        help="length of the randomized path to create and inject",
        type=int,
    )


    cfg_group = parser.add_argument_group(
        'Advanced config overrides',
        'The following allow overriding default values from config/defaults.py'
    )
    config.arg_parse_config_vars(cfg_group)

    prog_args = vars(parser.parse_args())
    # the following were aliased above, make sure they're updated globally
    prog_args.update({
        'SPARQL_ENDPOINT': prog_args['sparql_endpoint'],
        'GT_ASSOCIATIONS_FILENAME': prog_args['gtps_filename'],
    })
    config.finalize(prog_args)

    return prog_args


def _main(sparql, gtps):
    from gp_learner import find_graph_pattern_coverage
    from exception import GPLearnerTestPatternFoundException

    pattern_found = False
    try:
        res = find_graph_pattern_coverage(sparql, gtps)
        result_patterns, coverage_counts, gtp_scores = res
        if gtp_scores.remaining_gain < 5 * config.MIN_REMAINING_GAIN:
            # 5*: in rare cases happens that some of the gtps run into a timeout
            pattern_found = True
    except GPLearnerTestPatternFoundException:
        # XXX: unused at the moment due to quick stop already taking care
        # might override user_callback_per_generation
        pattern_found = True
    return pattern_found


def main(**kwds):
    from eval.random_path_loader import path_loader
    from ground_truth_tools import get_semantic_associations
    from utils import log_all_exceptions
    from utils import curify

    logging.info('encoding check: äöüß🎅')  # logging utf-8 byte string
    logging.info(u'encoding check: äöüß\U0001F385')  # logging unicode string
    logging.info(u'encoding check: äöüß\U0001F385'.encode('utf-8'))  # convert
    print('encoding check: äöüß🎅')  # printing utf-8 byte string
    print(u'encoding check: äöüß\U0001F385')  # printing unicode string


    # inject triples for a random path of given length into endpoint
    eval_gp = path_loader(**kwds)


    sparql_endpoint = kwds['sparql_endpoint']
    gtps_filename = kwds['gtps_filename']
    length = kwds['length']

    gtps = tuple(sorted(
        get_semantic_associations(gtps_filename)))
    # for s, t in gtps:
    #     print(curify(s))
    #     print(curify(t))
    #     print('')

    sparql = SPARQLWrapper.SPARQLWrapper(sparql_endpoint)

    tic = datetime.utcnow()
    # noinspection PyBroadException
    try:
        pattern_found = log_all_exceptions(logger)(_main)(sparql, gtps)
        return_code = 0 if pattern_found else 1
        tac = datetime.utcnow()
        logger.info(
            "search for pattern took %s and was %s",
            tac - tic,
            'successful' if pattern_found else 'unsuccessful'
        )
    except Exception:
        tac = datetime.utcnow()
        logger.exception(
            "search for pattern took %s and was aborted due to exception",
            tac - tic,
        )
        return_code = 2

    # return code's 0 is success, turn into more intuitive encoding for file
    res = {0: 1, 1: 0, 2: -1}[return_code]

    fn = path.join(config.RESDIR, RESULT_FILENAME)
    with open(make_dirs_for(fn), 'a') as f:
        f.write(
            'len: %d, result: %d, took: %.1f s, end (UTC): %s\n'
            'eval %s\n\n' % (
                length, res, timedelta_to_s(tac - tic), datetime.utcnow(),
                eval_gp
            )
        )
    sys.exit(return_code)


if __name__ == "__main__":
    logger.info('init run: origin')
    import config
    prog_kwds = parse_args()
    main(**prog_kwds)
else:
    logger.info('init run: worker')
