#!/usr/bin/env bash
# Will run gp_learner in 3 steps, running visualise/prepare and putting all of
# the files in the specified folder
set -e
set -u
set -o pipefail
# set -x

function usage() {
    echo "usage: $0 bundle_rel_path http://sparqlendpoint [args_for_run.py]"
    exit 1
}

if [[ $# -lt 2 ]]; then
    usage
fi
bundle="$1"
shift
sparql="$1"
shift

cd "$(dirname $0)"

export PYTHONIOENCODING=utf-8

mkdir "$bundle" || read -p "proceeding will overwrite some files! [ENTER/CTRL+C]"
echo "Runtime options:
sparql_endpoint: $sparql
bundle: $bundle
other: $@
" > "$bundle"/bundle_runtime_options.log

./clean_logs.sh

echo "training"
python -m scoop -n8 run.py --sparql_endpoint="$sparql" --predict='' "$@" 2>&1 | tee >(gzip > "$bundle"/train.log.gz)

echo "preparing visualise"
pushd visualise
python prepare.py
popd
echo "copying visualise to $bundle"
rsync -aPui visualise/data visualise/static visualise/visualise.html "$bundle"/visualise/

echo "predict train set"
python -m scoop -n8 run.py --sparql_endpoint="$sparql" --predict='train_set' "$@" 2>&1 | tee >(gzip > "$bundle"/predict_train.log.gz)

echo "predict test set"
python -m scoop -n8 run.py --sparql_endpoint="$sparql" --predict='test_set' "$@" 2>&1 | tee >(gzip > "$bundle"/predict_test.log.gz)

echo "moving training results to $bundle"
mv results/results* results/top_graph_patterns* "$bundle/"

echo "done, bundle size:"
du -sh "$bundle"

ls logs/*warning* 2> /dev/null || true
if ls logs/*error* 2> /dev/null ; then
    cp -a logs/*error* "$bundle/"
    exit 2
fi
