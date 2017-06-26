#!/bin/bash
#set -x
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
# for i in data/*.csv* ; do gt=${i%%.*} ; gt=${gt#*/} ; echo $gt ; sbatch -J "$gt" --array=1-10 -N1 -n1 --tasks-per-node=1 -c16 --mem 60G --tmp 70G -t12:00:00 -D $SCRATCH/gp_learner/logs gp_learner.sh --virtuoso_db_pack=$SCRATCH/bigfiles/dbpedia_ext_virtuoso_db.tar.lzop $SCRATCH/gp_learner/results/${gt}_$(date '+%Y-%m-%d') --NGEN=32 --POPSIZE=1000 --associations_filename=$i ; done

SPARQL="http://localhost:8890/sparql"
PROCESSES=${SLURM_CPUS_PER_TASK}
PROCESSES=${PROCESSES:-$SLURM_CPUS_ON_NODE}
PROCESSES=${PROCESSES:-16}
PROCESSES=$(( $PROCESSES * 3 / 4 ))  # leave some for virtuoso
VIRTUOSO_MAX_MEM=${VIRTUOSO_MAX_MEM:-40000000}  # in KB, don't ask why (should leave enough room for gp learner to 60 GB)
VIRTUOSO_INI="${VIRTUOSO_INI:-$HOME/virtuoso.ini}"
BUNDLE_POST=${BUNDLE_POST:-}

function usage() {
    echo "usage: $0 [--virtuoso_db_pack=/virtuso_db.tar.lzop] [--sparql_endpoint=$SPARQL] [--processes=$PROCESSES] [--visualise] [--] bundle_path [args_for_run.py]" >&2
    echo
    echo "--sparql_endpoint defaults to $SPARQL"
    echo "--processes defaults to slurm specs or $PROCESSES"
    exit 1
}


function watch_resource_usage() {
    set +x
    echo "watcher pid $$, $BASHPID, ppid $PPID"
    low_load_counter=0
    while true ; do
        echo -e "\nresource usage on host: $(hostname) working on $bundle"
        top -n1 -b -o'%CPU' | head -n12 || true
        top -n1 -b -o'%MEM' | head -n12 | tail -n+6 || true
        df -h "$bundle" "$TMPDIR" 2>/dev/null || true
        virtuoso_setup=$(pgrep -f virtuoso_unpack_local)
        load=$(uptime | sed -n -e 's/^.*load average: .*, \(.*\), .*$/\1/p')
        if [[ -z "$virtuoso_setup" && $(echo "$load < 1" | bc) -eq 1 ]] ; then
            # it seems that when a scoop worker is killed due to out of mem, the
            # parent process locks up waiting for its answer :(
            echo "5 min load avg. is below 1..."
            low_load_counter=$(($low_load_counter + 1))
            if [[ "$low_load_counter" -ge 5 ]] ; then
                echo "killing main script's sub-processes"
                pkill -P "$$" || true
                sleep 60
                # cleanup should stop us here
                echo "killing main script"
                kill "$$" || true
                sleep 60
                echo "killing parent's ($PPID) sub-processes"
                pkill -P "$PPID" || true
                sleep 60
                echo "killing parent $PPID"
                kill "$PPID" || true
                sleep 15
                echo "kill -9 $$ ?!?"
                kill -9 "$$" || true
                sleep 15
                echo "kill -9 $PPID ?!?"
                kill -9 "$PPID" || true
                sleep 15
                # cleanup should really have stopped us long before this
                echo "giving up, can't terminate"
            fi
        else
            low_load_counter=0
        fi
        sleep ${1:-300}
    done
}

function file_roll() {
    fn="$1"
    ext="$2"
    if [[ -e "$fn.$ext" ]] ; then
        i=1
        while [[ -e "$fn.$i.$ext" ]] ; do
            ((i++))
        done
        fn="$fn.$i"
    fi
    touch "$fn.$ext"
    echo "$fn.$ext"
}

function time_echo() {
    echo -n "$@" ;
    date --rfc-3339=seconds 2>/dev/null || date '+%F %T'
}

if [[ ! -f run.py || ! -f gp_learner.py ]] ; then
    echo "should be invoked from gp_learner dir, trying to change into it..." >&2
    pwd >&2
    echo "$0" >&2
    echo "$@" >&2
    env >&2
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

if [[ $# -lt 1 ]] ; then
    echo "no bundle dir specified?" >&2
    usage
fi


# slurm support (cluster)
if [[ -n "$SLURM_JOB_ID" ]] ; then
    # scoop's slurm host parsing fails and we want to run on one only anyhow...
    host="--host $(hostname)"
    if [[ -n "$SLURM_ARRAY_TASK_ID" ]] ; then
        bundle="$1/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}_on_${SLURM_JOB_NODELIST}"
    else
        bundle="$1/${SLURM_JOB_ID}_on_${SLURM_JOB_NODELIST}"
    fi
else
    host=""
    bundle="$1"
fi
bundle="$bundle$BUNDLE_POST"
shift


if [[ -d "venv" ]] ; then
    . venv/bin/activate
else
    echo "WARNING: could not find virtualenv, trying to run with current env"
fi


function cleanup_gp_learner() {
    if [[ -n "$resource_watcher_pid" ]] ; then
        kill "$resource_watcher_pid" || true
    fi

    if [[ -n "$virtuoso_watchdog_pid" ]] ; then
        kill "$virtuoso_watchdog_pid" || true
    fi

    if [[ -n "$VIRTUOSO_DB_PACK" ]] ; then
        isql <<< "shutdown;" || true
    fi

    ls "$GP_LEARNER_LOG_DIR"/*warning* 2> /dev/null || true
    if ls "$GP_LEARNER_LOG_DIR"/*error* 2> /dev/null ; then
        if [[ -n "$SLURM_JOB_ID" ]] ; then
            tar -cf "$bundle/error_logs.tar" "$GP_LEARNER_LOG_DIR"/*error*
        fi
        exit 2
    fi

    # wait for virtuoso to actually shut down...
    for i in {1..18} ; do
        pgrep virtuoso > /dev/null && break || sleep 10
    done
}
trap cleanup_gp_learner EXIT


function virtuoso_watchdog() {
    # gracefully restarts virtuoso if it consumes too much memory
    set +x
    echo "watching virtuoso memory < $VIRTUOSO_MAX_MEM ..."
    while true ; do
        if [[ -z "$bundle" ]] ; then sleep 5 ; continue ; fi
        # get virtuoso memory
        virtuoso_pid=$(pgrep virtuoso)
        if [[ -n "$virtuoso_pid" ]] ; then
            virtuoso_mem=$(ps h -o rss "$virtuoso_pid")
            if [[ "$virtuoso_mem" -gt "$VIRTUOSO_MAX_MEM" ]] ; then
                # we're above the maximum memory that virtuoso should use
                echo "asking gp learner to pause..."
                touch "$bundle/results/pause.lck"
                while [[ -z $(cat "$bundle/results/pause.lck") ]] ; do sleep 5 ; done
                echo "gp learner paused, stopping virtuoso..."
                isql <<< "shutdown;"
                while pgrep virtuoso ; do sleep 5 ; done
                echo "virtuoso stopped, starting again..."
                scripts/virtuoso_unpack_local_and_run.sh "$VIRTUOSO_DB_PACK" "$VIRTUOSO_INI" >&2
                echo "ok, virtuoso back up, removing pause.lck"
                rm "$bundle/results/pause.lck"
                echo "done, thanks for flying with WTF!!!"
            fi
        else
            echo "virtuoso not running? trying to start it..."
            scripts/virtuoso_unpack_local_and_run.sh "$VIRTUOSO_DB_PACK" "$VIRTUOSO_INI" >&2
        fi
        sleep 60
    done
}

if [[ -n "$SLURM_JOB_ID" ]] ; then
    echo "script pid $$, ppid $PPID" >&2
    watch_resource_usage >&2 &
    resource_watcher_pid=$!
fi

if [[ -n "$VIRTUOSO_DB_PACK" ]] ; then
    echo "disk free before virtuoso db unpacking" >&2
    df -h >&2
    scripts/virtuoso_unpack_local_and_run.sh "$VIRTUOSO_DB_PACK" "$VIRTUOSO_INI" >&2
    echo "disk free after virtuoso db unpacking" >&2
    df -h >&2

    virtuoso_watchdog >&2 &
    virtuoso_watchdog_pid=$!
fi



mkdir -p "$bundle/results"

bundle_log="$bundle/bundle_summary.log"
if [[ -f "$bundle_log" ]] ; then
    echo "continuing previous training" >&2
    echo -e "\n\n" >> "$bundle_log"
fi
echo "Runtime options:
version: $(git log -n1 --oneline --date=iso --pretty=format:'%h - %s (%cd)')
processes: $PROCESSES
sparql_endpoint: $SPARQL
bundle: $bundle
other: $@
" | tee -a "$bundle_log"

time_echo "start: " | tee -a "$bundle_log"

# if running on slurm cluster, write logs locally and only write back on error (see cleanup)
if [[ -n "$SLURM_JOB_ID" && -n "$TMPDIR" ]] ; then
    export GP_LEARNER_LOG_DIR="$TMPDIR/logs"
else
    mkdir -p "$bundle/logs"
    export GP_LEARNER_LOG_DIR="$bundle/logs"
fi

export PYTHONIOENCODING=utf-8


time_echo "training start: " | tee -a "$bundle_log"
logfile="$(file_roll "$bundle/train.log" gz)"
python -m scoop $host -n${PROCESSES} run.py --sparql_endpoint="$SPARQL" --RESDIR="$bundle/results" --predict='' "$@" 2>&1 | tee >( gzip > "$logfile")
time_echo "training end: " | tee -a "$bundle_log"

time_echo "predict train set start: " | tee -a "$bundle_log"
logfile="$(file_roll "$bundle/predict_train.log" gz)"
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scoop $host -n${PROCESSES} run.py --sparql_endpoint="$SPARQL" --RESDIR="$bundle/results" --predict='train_set' "$@" 2>&1 | tee -i >(gzip > "$logfile")
time_echo "predict train set end: " | tee -a "$bundle_log"

time_echo "predict test set start: " | tee -a "$bundle_log"
logfile="$(file_roll "$bundle/predict_test.log" gz)"
python -m scoop $host -n${PROCESSES} run.py --sparql_endpoint="$SPARQL" --RESDIR="$bundle/results" --predict='test_set' "$@" 2>&1 | tee -i >(gzip > "$logfile")
time_echo "predict test set end: " | tee -a "$bundle_log"

if [[ $VISUALISE = true ]] ; then
    time_echo "preparing visualise start: " | tee -a "$bundle_log"
    python visualise/prepare.py -i "$bundle/results" -o "$bundle/visualise"
    time_echo "preparing visualise end: " | tee -a "$bundle_log"
fi

time_echo "repacking top_graph_patterns: " | tee -a "$bundle_log"
scripts/pack_top_graph_patterns.sh "$bundle/results/generations" || true

echo "done, bundle size:"
du -sh "$bundle"

time_echo "end: " | tee -a "$bundle_log"


