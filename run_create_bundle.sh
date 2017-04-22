#!/bin/bash
set -x
set -e
set -o pipefail

# example invocations:
# # minimal:
# ./run_create_bundle.sh results/foo
#
# # to also create html visualisation:
# ./run_create_bundle.sh --visualise results/foo_$(date '+%Y-%m-%d')
#
# # non default sparql endpoint:
# ./run_create_bundle.sh --sparql_endpoint="http://ip:port/sparql" results/foo_$(date '+%Y-%m-%d')
#
# # to set up and run against a local virtuoso SPARQL endpoint from DB pack
# ./run_create_bundle.sh --virtuoso_db_pack=/virtuoso_db.tar.lzop results/foo_$(date '+%Y-%m-%d')
#
# # to specify args for gp_learner
# ./run_create_bundle.sh results/foo --NGEN=32 --POPSIZE=1000
#
# # to run on slurm cluster (each gt file in an array job 10 times):
# for i in data/*.csv data/*.csv.gz ; do gt=${i%%.*} ; gt=${gt#*/} ; echo $gt ; sbatch -J "$gt" --array=1-10 -N1 -n1 --tasks-per-node=1 -c16 --mem 64G --tmp 70G -t12:00:00 -D $SCRATCH/gp_learner/logs gp_learner.sh --virtuoso_db_pack=$SCRATCH/bigfiles/dbpedia_ext_virtuoso_db.tar.lzop $SCRATCH/gp_learner/results/${gt}_$(date '+%Y-%m-%d') --NGEN=32 --POPSIZE=1000 --associations_filename=$i ; done


SPARQL="http://localhost:8890/sparql"
PROCESSES=${SLURM_CPUS_PER_TASK}
PROCESSES=${PROCESSES:-SLURM_CPUS_ON_NODE}
PROCESSES=${PROCESSES:-8}


function usage() {
    echo "usage: $0 [--virtuoso_db_pack=/virtuso_db.tar.lzop] [--sparql_endpoint=$SPARQL] [--processes=$PROCESSES] [--visualise] [--] bundle_path [args_for_run.py]" >&2
    echo
    echo "--sparql_endpoint defaults to $SPARQL"
    echo "--processes defaults to slurm specs or $PROCESSES"
    exit 1
}


function watch_resource_usage() {
    set +x
    secs=${1:-120}
    while true ; do
        h=$(hostname)
        echo "resource usage on host: $h"
        top -n1 -b -o'%CPU' | head -n12
        top -n1 -b -o'%MEM' | tail -n+6 | head -n6
        sleep $secs
    done
}

function time_echo() {
    echo -n "$@" ;
    date --rfc-3339=seconds
}

if [[ ! -d venv || ! -f run.py || ! -f gp_learner.py ]] ; then
    echo "should be invoked from gp_learner dir, trying to change into it..."
    pwd
    echo "$0"
    echo "$@"
    # env
    if [[ -n "$SLURM_SUBMIT_DIR" ]] ; then
        cd "$SLURM_SUBMIT_DIR"
    else
        cd "$(dirname $0)"
    fi
fi

# argparsing ...
for arg in "$@" ; do
    case "$arg" in
        --virtuoso_db_pack=*)
            VIRTUOSO_DB_PACK="${arg#*=}"
            if [[ ! -f "$VIRTUOSO_DB_PACK" ]]; then
                echo "could not find virtuoso db pack: $VIRTUOSO_DB_PACK" >&2
                exit 2
            fi
            shift
            ;;
        --sparql_endpoint=*)
            SPARQL="${arg#*=}"
            shift
            ;;
        --processes=*)
            PROCESSES="${arg#*=}"
            shift
            ;;
        --visualise)
            VISUALISE=true
            shift
            ;;
        --)
            # done parsing args for us
            shift
            break
            ;;
        *)
            if [[ "$arg" =~ --.* ]] ; then
                echo "unknown arg $arg" >&2
                usage
            else
                # done parsing args for us
                break
            fi
            ;;
    esac
done


# slurm support (cluster)
if [[ -n $SLURM_JOB_ID ]] ; then
    if [[ -n $SLURM_ARRAY_TASK_ID ]] ; then
        bundle="$1/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_on_${SLURM_JOB_NODELIST}"
    else
        bundle="$1/${SLURM_JOB_ID}_on_${SLURM_JOB_NODELIST}"
    fi
else
    bundle="$1"
fi
shift



. venv/bin/activate


function cleanup_gp_learner() {
    if [[ -n $resource_watcher_pid ]] ; then
        kill "$resource_watcher_pid"
    fi

    if [[ -n $VIRTUOSO_DB_PACK ]] ; then
        isql <<< 'shutdown;'
    fi
}
trap cleanup_gp_learner EXIT

watch_resource_usage >&2 & resource_watcher_pid=$!

if [[ -n $VIRTUOSO_DB_PACK ]] ; then
    echo "disk free before virtuoso db unpacking"
    df -h >&2
    scripts/virtuoso_unpack_local_and_run.sh "$VIRTUOSO_DB_PACK" $HOME/virtuoso.ini >&2
    echo "disk free after virtuoso db unpacking"
    df -h >&2
fi



mkdir -p "$bundle/results"

bundle_log="$bundle/bundle_runtime_options.log"
echo "Runtime options:
processes: $PROCESSES
sparql_endpoint: $SPARQL
bundle: $bundle
other: $@
" > "$bundle_log"

time_echo "start: " | tee >> "$bundle_log"

# if running on slurm cluster, write logs locally and only write back on error (see end)
if [[ -n $SLURM_JOB_ID && -n $TMPDIR ]] ; then
    export GP_LEARNER_LOG_DIR="$TMPDIR/logs"
else
    mkdir -p "$bundle/logs"
    export GP_LEARNER_LOG_DIR="$bundle/logs"
fi

export PYTHONIOENCODING=utf-8


time_echo "training start: " | tee >> "$bundle_log"
python -m scoop --host $(hostname) -n${PROCESSES} run.py --sparql_endpoint="$SPARQL" --RESDIR="$bundle/results" --predict='' "$@" 2>&1 | tee >(gzip > "$bundle"/train.log.gz)
time_echo "training end: " | tee >> "$bundle_log"

time_echo "predict train set start: " | tee >> "$bundle_log"
python -m scoop --host $(hostname) -n${PROCESSES} run.py --sparql_endpoint="$SPARQL" --RESDIR="$bundle/results" --predict='train_set' "$@" 2>&1 | tee >(gzip > "$bundle"/predict_train.log.gz)
time_echo "predict train set end: " | tee >> "$bundle_log"

time_echo "predict test set start: " | tee >> "$bundle_log"
python -m scoop --host $(hostname) -n${PROCESSES} run.py --sparql_endpoint="$SPARQL" --RESDIR="$bundle/results" --predict='test_set' "$@" 2>&1 | tee >(gzip > "$bundle"/predict_test.log.gz)
time_echo "predict test set end: " | tee >> "$bundle_log"

if [[ $VISUALISE = true ]] ; then
    time_echo "preparing visualise start: " | tee >> "$bundle_log"
    python visualise/prepare.py -i "$bundle/results" -o "$bundle/visualise"
    time_echo "preparing visualise end: " | tee >> "$bundle_log"
fi


echo "done, bundle size:"
du -sh "$bundle"

ls "$GP_LEARNER_LOG_DIR"/*warning* 2> /dev/null || true
if ls "$GP_LEARNER_LOG_DIR"/*error* 2> /dev/null ; then
    if [[ -n $SLURM_JOB_ID ]] ; then
        tar -cf "$bundle/error_logs.tar" "$GP_LEARNER_LOG_DIR"/*error*
    fi
    exit 2
fi

time_echo "end: " | tee >> "$bundle_log"


