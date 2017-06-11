#!/usr/bin/env bash

usage() {
    echo "usage: $0 generations" >&2
    exit 1
}

if [[ $# -ne 1 ]] ; then
    usage
fi

gen_folder="$1"
if [[ ! -d "$gen_folder" ]] ; then
    return
fi

if command -v pxz > /dev/null ; then
    # properly pack top_graph_pattern run files together
    size_before=$(du -sh "$gen_folder")
    pushd "$gen_folder" > /dev/null
    tgprs=$(
        ls top_graph_patterns_run*.json* |
        grep -Eo 'top_graph_patterns_run_[0-9]+_' |
        sort | uniq
    )
    for tgpr in $tgprs ; do
        ls $tgpr*.json* | xargs -t -P0 -n1 gunzip &&
        tar -cvf - $tgpr*.json | pxz > "${tgpr}pack.tar.xz" &&
        rm $tgpr*.json ||
        gzip $tgpr*.json  # re-pack in case something goes wrong
    done
    popd > /dev/null
    size_after=$(du -sh "$gen_folder")
    echo "size before top_graph_pattern packing:"
    echo "$size_before"
    echo "size after top_graph_pattern packing:"
    echo "$size_after"
fi
