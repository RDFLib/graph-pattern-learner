# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
        '--sparql_endpoint',
        help="the SPARQL endpoint to query",
        action="store",
        default=config.SPARQL_ENDPOINT,
    )

    gt_auto_split = parser.add_argument_group(
        "Ground Truth Auto Split",
        "Provide a single ground truth file that we automatically split into "
        "train- and test-set according to specified splitting variant."
    )
    gt_auto_split.add_argument(
        "--associations_filename", "--ground_truth_filename",
        help="ground truth source target file used for training and evaluation",
        action="store",
        default=config.GT_ASSOCIATIONS_FILENAME,
    )
    gt_auto_split.add_argument(
        "--splitting_variant",
        help="how to split the train, validation & test set (default: random)",
        action="store",
        default="random",
        choices=config.SPLITTING_VARIANTS,
    )

    gt_manual_split = parser.add_argument_group(
        "Ground Truth Manual Split",
        "Provide individual source target pair files for training and testing. "
        "If only one is given, make sure that --predict is set accordingly."
    )
    gt_train_filename = gt_manual_split.add_argument(
        "--train_filename",
        help="file with source target pairs for training",
        action="store",
        default=None,
    )
    gt_test_filename = gt_manual_split.add_argument(
        "--test_filename",
        help="file with source target pairs for testing",
        action="store",
        default=None,
    )

    parser.add_argument(
        "--print_train_test_sets",
        help="prints the sets used for training and testing",
        action="store",
        default=True,
        type=config.str_to_bool,
    )

    parser.add_argument(
        "--reset",
        help="remove previous training's result files if existing (otherwise "
             "the previous training's model will be loaded. If the training "
             "wasn't complete, its last completed run will be loaded and "
             "training will continue)",
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

    parser.add_argument(
        "--fusion_methods",
        help="Which fusion methods to train / use. During prediction, each of "
             "the learned patterns can generate a list of target candidates. "
             "Fusion allows to re-combine these into a single ranked list of "
             "predicted targets. By default this will train and use all "
             "implemented fusion methods. Any of them, or a ',' delimited list "
             "can be used to reduce the output (just make sure you ran "
             "--predict=train_set on them before). Also supports 'basic' and "
             "'classifier' as shorthands.",
        action="store",
        type=str,
        default=None,
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
        'GT_ASSOCIATIONS_FILENAME': prog_args['associations_filename'],
    })
    config.finalize(prog_args)


    from gp_learner import main
    main(**prog_args)
else:
    logger.info('init run: worker')
