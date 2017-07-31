# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys
from datetime import datetime
from functools import wraps
from os import path

import SPARQLWrapper
from cachetools import LFUCache
from rdflib import URIRef
from splendid import make_dirs_for
from splendid import timedelta_to_s

from flask import Flask
from flask import abort
from flask import jsonify
from flask import request
from flask_cors import CORS

# noinspection PyUnresolvedReferences
import logging_config

# not all import on top due to scoop and init...

logger = logging.getLogger(__name__)
app = Flask(__name__)
CORS(app)


@app.route("/api/ping", methods=["GET"])
def ping():
    return jsonify({
        'success': True
    })


@app.route("/api/graph_patterns", methods=["GET"])
def graph_patterns():
    global GPS_DICT
    if not GPS_DICT:
        GPS_DICT = {
            'graph_patterns': [
                {
                    k: v
                    for k, v in gp.to_dict().items()
                    if k in (
                        'fitness',
                        'fitness_weighted',
                        'fitness_description',
                        'sparql',
                        'graph_triples',
                        # 'matching_node_pairs',
                        # 'gtp_precisions',
                        'prefixes',
                    )
                }
                for gp in GPS
            ],
        }
    return jsonify(GPS_DICT)


@app.route("/api/predict", methods=["POST"])
def predict():
    source = request.form.get('source')
    # logger.info(request.data)
    # logger.info(request.args)
    # logger.info(request.form)
    if not source:
        abort(400, 'no source given')
    logger.info('predicting: %s', source)
    source = URIRef(source)

    return jsonify(PREDICT_CACHE[source])


def _predict(source):
    from fusion import fuse_prediction_results
    from gp_learner import predict_target_candidates
    from gp_query import calibrate_query_timeout

    timeout = TIMEOUT if TIMEOUT > 0 else calibrate_query_timeout(SPARQL)
    gp_tcs = predict_target_candidates(SPARQL, timeout, GPS, source)
    fused_results = fuse_prediction_results(
        GPS,
        gp_tcs,
        FUSION_METHODS
    )
    orig_length = max([len(v) for k, v in fused_results.items()])
    if MAX_RESULTS > 0:
        for k, v in fused_results.items():
            del v[MAX_RESULTS:]
    mt = MAX_TARGET_CANDIDATES_PER_GP
    if mt < 1:
        mt = None
    # logger.info(gp_tcs)
    res = {
        'source': source,
        'orig_result_length': orig_length,
        'graph_pattern_target_candidates': [sorted(tcs[:mt]) for tcs in gp_tcs],
        'fused_results': fused_results,
    }
    return res


@app.route("/api/feedback", methods=["POST"])
def feedback():
    # TODO: add timestamps, ips, log to different file
    fb = {
        'source': request.form.get('source'),
        'target': request.form.get('target'),
        'feedback': request.form.get('feedback') == 'true',
        'fusion_method': request.form.get('fusion_method'),
        'rank': int(request.form.get('rank')),
    }
    logger.info('received feedback: %s', json.dumps(fb))
    res = {
        'success': True,
        'msg': 'thanks ;)',
    }
    return jsonify(res)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description='gp learner prediction model server',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # flask settings
    parser.add_argument(
        "--host",
        help="listen on IP",
        action="store",
        default="0.0.0.0",
    )
    parser.add_argument(
        "--port",
        help="port to listen on",
        action="store",
        default="8080",
    )
    parser.add_argument(
        "--flask_debug",
        help="flask debug mode",
        action="store_true",
        default=False,
    )

    # gp learner settings
    parser.add_argument(
        "--resdir",
        help="result directory of the model to serve (overrides --RESDIR)",
        action="store",
        required=True,
    )

    parser.add_argument(
        "--sparql_endpoint",
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

    # serve specific configs
    parser.add_argument(
        "--timeout",
        help="sets the timeout in seconds for each query (0: auto calibrate)",
        action="store",
        type=float,
        default=.5,
    )
    parser.add_argument(
        "--max_results",
        help="limits the result list lengths to save bandwidth (0: no limit)",
        action="store",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--max_target_candidates_per_gp",
        help="limits the target candidate list lengths to save bandwidth "
             "(0: no limit)",
        action="store",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--predict_cache_size",
        help="how many prediction results to cache",
        action="store",
        type=int,
        default=1000,
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
        'RESDIR': prog_args['resdir'],
    })
    config.finalize(prog_args)

    return prog_args


def init(**kwds):
    from gp_learner import main
    return main(**kwds)


if __name__ == "__main__":
    logger.info('init run: origin')
    import config
    prog_kwds = parse_args()
    SPARQL, GPS, FUSION_METHODS = init(**prog_kwds)

    TIMEOUT = prog_kwds['timeout']
    MAX_RESULTS = prog_kwds['max_results']
    MAX_TARGET_CANDIDATES_PER_GP = prog_kwds['max_target_candidates_per_gp']
    GPS_DICT = None
    PREDICT_CACHE = LFUCache(prog_kwds['predict_cache_size'], _predict)
    if prog_kwds['flask_debug']:
        logger.warning('flask debugging is active, do not use in production!')
    app.run(
        host=prog_kwds['host'],
        port=prog_kwds['port'],
        debug=prog_kwds['flask_debug'],
    )
else:
    logger.info('init run: worker')
