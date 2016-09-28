# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
logger = logging.getLogger(__name__)

# noinspection PyUnresolvedReferences
import logging_config

if __name__ == "__main__":
    logger.info('init run: origin')

    import argparse
    import config

    parser = argparse.ArgumentParser(
        description='learn graph patterns',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--loglevel",
        help="desired log level",
        action="store",
        default=logging.INFO,
    )

    parser.add_argument(
        '--sparql_endpoint',
        help="the SPARQL endpoint to query",
        action="store",
        default=config.SPARQL_ENDPOINT,
    )

    parser.add_argument(
        "--associations_filename",
        help="ground truth source target file used for training and evaluation",
        action="store",
        default=config.GT_ASSOCIATIONS_FILENAME,
    )

    parser.add_argument(
        "--print_train_test_sets",
        help="prints the sets used for training and testing",
        action="store",
        default=True,
        type=config.str_to_bool,
    )

    parser.add_argument(
        "--splitting_variant",
        help="how to split the train, validation & test set",
        action="store",
        default="random",
        choices=config.SPLITTING_VARIANTS,
    )

    parser_learn_group = parser.add_mutually_exclusive_group()
    parser_learn_group.add_argument(
        "--learn_patterns",
        help="(re-)learn patterns for given list from SPARQL endpoint",
        action="store_true",
        default=False,
    )

    parser_learn_group.add_argument(
        "--learn_patterns_resume",
        help="if unfinished continues learning patterns from the last complete "
             '"run" of a previous invocation',
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--print_topn_raw_patterns",
        help="how many of the found (raw, unclustered) patterns to print out",
        action="store",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--print_edge_only_connected_patterns",
        help="separate print out of edge only connected and mixed var patterns",
        action="store",
        default=True,
        type=config.str_to_bool,
    )

    parser.add_argument(
        "--show_precision_loss_by_query_reduction",
        help="shows a plot of expected precision degradation of prediction "
             "(on the training set) if the amount of queries per prediction is "
             "limited (max_q).",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--max_queries",
        help="limits the amount of queries per prediction (0: no limit)",
        action="store",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--clustering_variant",
        help="if specified use this clustering variant for query reduction, "
             "otherwise select the best from various.",
        action="store",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--print_query_patterns",
        help="print the graph patterns which are used to make predictions",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--predict",
        help="evaluate the learned patterns by predicting the targets for the "
             "sources in the specified set of ground truth source-target-pairs "
             "and comparing the predicted targets to the ground truth targets. "
             "During development of this algorithm, parameter tuning and to "
             "get an upper bound use 'train_set', finally use 'test_set'. To "
             "disable evaluation set to ''.",
        action="store",
        type=str,
        choices=("test_set", "train_set", "manual", ""),
        default="",
    )

    cfg_group = parser.add_argument_group(
        'Advanced config overrides',
        'The following allow overriding default values from config/defaults.py'
    )
    config.arg_parse_config_vars(cfg_group)

    prog_args = vars(parser.parse_args())
    config.finalize(prog_args)


    from gp_learner import main
    main(**prog_args)
else:
    logger.info('init run: worker')
