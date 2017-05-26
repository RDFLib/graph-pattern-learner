#! /bin/bash
set -e
set -x

usage() {
    echo "usage: $0 result_folder" >&2
    exit 1
}

if [[ $# -ne 1 ]] ; then
    usage
fi

res="$1"
mkdir "$res"/runs "$res"/generations || true
mv "$res"/results_run_* "$res"/runs/ || true
mv "$res"/top_graph_patterns_run_* "$res"/generations/ || true

l="$res/top_graph_patterns_current.json.gz"
t="$(readlink "$l")"
if [[ ! $t == $res/generations/* ]] ; then
    rm $l
    ln -s "$res/generations/$t" "$l"
fi

l="$res/results_current.json.gz"
t="$(readlink "$l")"
if [[ ! $t == $res/runs/* ]] ; then
    rm $l
    ln -s "$res/runs/$t" "$l"
fi
