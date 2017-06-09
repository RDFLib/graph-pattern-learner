#!/usr/bin/env bash

usage() {
    echo "usage: $0 generations" >&2
    exit 1
}

if [[ $# -ne 1 ]] ; then
    usage
fi

gen_folder="$1"

if command -v pxz > /dev/null ; then
    # properly pack top_graph_pattern run files together
    pushd "$gen_folder" > /dev/null
    tgprs=$(
        ls top_graph_patterns_run*.json* |
        grep -Eo 'top_graph_patterns_run_[0-9]+_' |
        sort | uniq
    )
    for tgpr in $tgprs ; do
        ls $tgpr*.json* | xargs -t -P0 -n1 gunzip &&
        tar -cvf - $tgpr*.json | pxz > "${tgpr}pack.tar.xz" ||
        gzip $tgpr*.json  # re-pack in case something goes wrong
    done
    popd > /dev/null
fi
