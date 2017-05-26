#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
from os import path
import re
import shutil
from distutils.dir_util import copy_tree
import time
from collections import defaultdict
import base64
from urllib import quote
import json
from glob import glob
import gzip
from subprocess import Popen, PIPE
from multiprocessing import Pool, cpu_count

import logging
logging.basicConfig(level=logging.DEBUG)

_script_dir = path.dirname(path.abspath(__file__))
IN_DIR = path.join(_script_dir, '..', 'results')
OUT_DIR = path.join(_script_dir, 'data')

TIMESTAMP_REGEX = r'\d\d\d\d-\d\d-\d\dT\d\d-\d\d-\d\d'

START_ID = "?source"
END_ID = "?target"

SPARQL_BASE_URI = "http://dbpedia.org/sparql?qtxt="


def get_all_runs_and_gens(src):
    """returns dict of type {run: gens, run: gens, ... }"""
    gens = defaultdict(lambda: 0)
    t = 'top_graph_patterns_run_*_gen_*.json.gz'
    fns = glob(path.join(src, t)) + glob(path.join(src, 'generations', t))
    for fn in fns:
        m = re.match(
            r".*/top_graph_patterns_run_(?P<run>\d+)_gen_(?P<gen>\d+).*"
            r"\.json\.gz",
            fn
        )
        if not m:
            raise IOError("No matching results!: " + fn)
        run = int(m.group("run"))
        gen = int(m.group("gen"))
        gens[run] = max(gens[run], gen)
    return dict(gens)


def generate_global_vars_js(src, dst):
    runsgens = get_all_runs_and_gens(src)
    max_run = max(runsgens.keys())
    current = 'top_graph_patterns_run_%02d_gen_%02d.json.gz' % (
        max_run, runsgens[max_run])
    js_target = path.join(dst, "global_vars.js")
    with open(js_target, "w") as f:
        f.write(
            "var RUNS_GENS_DICT = JSON.parse('%s');\n"
            "var START_FILENAME = '%s';\n"
            "var SCRIPT_ROOT = '%s';\n"
            "var SPARQL_BASE_URI = '%s';\n"
            % (json.dumps(runsgens), current, './data', SPARQL_BASE_URI)
        )
    return runsgens


def copy_single_latest_file_without_timestamp(src, dst, fn):
    fn_base = fn.rsplit('.json.gz', 1)[0]
    t = '%s_[0-9][0-9][0-9][0-9]-*.json.gz' % fn_base
    fns = glob(
        path.join(src, t)) + glob(
        path.join(src, 'runs', t)) + glob(
        path.join(src, 'generations', t))
    fns_ts = [
        fn_ for fn_ in fns
    ]
    fns_ts = sorted(fns_ts, key=path.basename)
    if not fns_ts:
        logging.warning("Couldn't find %s" % fn)
        return
    shutil.copy(fns_ts[-1], path.join(dst, fn))


def clear_target_folder(dst):
    logging.info("Clearing folder %s" % dst)
    for fn in glob(path.join(dst, '*.json.gz')):
        os.remove(fn)


def copy_latest_without_timestamp(runs_gens, src, dst):
    basenames = ['results.json.gz']
    for run in runs_gens:
        basenames.append('results_run_%02d.json.gz' % run)
        for gen in range(runs_gens[run] + 1):
            basenames.append('top_graph_patterns_run_%02d_gen_%02d.json.gz'
                             % (run, gen))
    for bn in basenames:
        copy_single_latest_file_without_timestamp(src, dst, bn)


def convert_content(fn, cont):
    res = []
    for p in cont['patterns']:
        gp = p['graph_pattern']
        nodes = []
        links = []
        for link in gp["graph_triples"]:
            for i in (0, 2):
                all_node_ids = [n["id"] for n in nodes]
                if not link[i] in all_node_ids:
                    nodes += [{
                        "id": link[i],
                        "label": link[i],
                        "start": link[i] == START_ID,
                        "end": link[i] == END_ID
                    }]
            link_id = '#'.join(
                map(base64.b64encode, [l.encode('utf-8') for l in link]))
            links += [{
                "id": link_id,
                "from": link[0],
                "to": link[2],
                "label": link[1]
            }]
        split_query = gp["sparql"].replace('%', '%%').split("{\n", 1)
        format_str = (
            split_query[0] +
            "{\n VALUES (?source) { (%(source)s) }\n" +
            split_query[1]
        )

        matching_node_pairs = [
            (
                source,
                target,
                (
                 # quote(format_str % {'source': source, 'target': target}))
                 quote((format_str % {'source': source}).encode('utf-8'),
                       safe=""))
            )
            for source, target in gp["matching_node_pairs"]
        ]
        res += [{
            "nodes": nodes,
            "links": links,
            "fitness": gp["fitness"],
            "fitness_description": gp["fitness_description"],
            "matching_node_pairs": matching_node_pairs,
            "gtp_precisions": gp["gtp_precisions"],
            "sparql_query": gp["sparql"],
            "sparql_link": quote(gp["sparql"].encode('utf-8'), safe="")
        }]
    res = {
        "graphs": res,
        "filename": path.basename(fn),
        "timestamp": cont["timestamp"],
        "ground_truth_pairs": cont["ground_truth_pairs"],
        "generation_number": cont.get("generation_number", -1),
        "run_number": cont.get("run_number", -1),
    }
    if cont.get("coverage_max_precision"):
        res["coverage_max_precision"] = cont["coverage_max_precision"]

    return res


def fast_gunzip(fname):
    """Quicker version of gzip.open() via system tools.

    Directly access the gzipped files a lot faster than gzip.open.
    """
    return Popen(['gunzip', '-c', fname], bufsize=2**15, stdout=PIPE).stdout


def fast_gzip(fname):
    with open(fname, 'w') as f:
        return Popen(['gzip'], bufsize=2**15, stdout=f, stdin=PIPE).stdin


def prepare_compressed_content(arg):
    i, n, fn = arg
    logging.debug("converting %03d / %03d: %s", i + 1, n, fn)
    with fast_gunzip(fn) as f:
        res = json.load(f)
    res = convert_content(fn, res)
    with fast_gzip(fn) as f:
        json.dump(res, f, indent=2)


def prepare_content_of_all_files(dst=OUT_DIR):
    fns = [fn_ for fn_ in sorted(glob(path.join(dst, '*.json.gz')))]
    n = len(fns)
    p = Pool(cpu_count())  # gzip processes
    p.map(
        prepare_compressed_content,
        [(i, n, fn) for i, fn in enumerate(fns)],
        chunksize=1
    )


def bundle_html(dst):
    if not dst == OUT_DIR:
        # not in the current folder, copy over html and static files
        pdst = path.dirname(dst)  # strip final '/data'
        logging.info('bundling visualise html files to %s', pdst)
        copy_tree(
            path.join(_script_dir, 'static'),
            path.join(pdst, 'static'))
        shutil.copy(path.join(_script_dir, 'visualise.html'), pdst)


def main(src=IN_DIR, target=OUT_DIR):
    t = time.time()
    if not target.endswith('data'):
        # make sure we have a data subfolder in target folder
        target = path.join(target, 'data')
    if not path.exists(target):
        os.makedirs(target)
    else:
        clear_target_folder(target)

    logging.info("Generating global_vars.json...")
    runs_gens = generate_global_vars_js(src, target)
    logging.info(runs_gens)

    logging.info("Copying & Renaming graph pattern files...")
    copy_latest_without_timestamp(runs_gens, src, target)

    logging.info("Updating json content...")
    prepare_content_of_all_files(target)

    bundle_html(target)

    logging.info("Done with it!")
    logging.info("It took %.2f seconds" % (time.time() - t))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='visualise learned patterns')

    parser.add_argument("-i", "--input",
                        help="the source folder to find the results in",
                        action="store", default=IN_DIR, dest="input")
    parser.add_argument("-o", "--output",
                        help="the target folder",
                        action="store", default=OUT_DIR, dest="output")

    args = parser.parse_args()

    main(args.input, args.output)
