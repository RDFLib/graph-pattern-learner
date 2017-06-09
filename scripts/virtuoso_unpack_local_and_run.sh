#! /bin/bash
set -e
set -x

TMPDIR="${TMPDIR:-/tmp/gp_learner_$RANDOM}"
TMP_DIR="$TMPDIR/virtuoso_db"
#PACKER="gzip"
PACKER="${PACKER:-lzop}"

if ! hash $PACKER 2>/dev/null ; then
    echo "packer $PACKER doesn't exist, aborting..." >&2
    exit 3
fi

usage() {
    echo "usage: $0 <virtuoso.tar.$PACKER> <virtuoso.ini>" >&2
    echo "" >&2
    echo "will unpack the given virtuoso db pack to $TMP_DIR" >&2
    echo "and start it with the given ini file" >&2
    exit 1
}

if [[ $# -ne 2 ]] ; then
    usage
fi

pack="$1"
ini="$2"

if [[ ! -f "$pack" ]] ; then
    echo "cannot find $pack, aborting..." >&2
    exit 3
fi

if [[ ! -f "$ini" ]] ; then
    echo "cannot find $ini, aborting..." >&2
    exit 3
fi

function start_db() {
    cd "$TMP_DIR"
    echo -n "starting db ... "
    date --rfc-3339=seconds
    virtuoso-t +wait +configfile "$ini"
    #virtuoso-t +foreground +configfile "$ini"
    echo -n "db up "
    date --rfc-3339=seconds

    echo "warmup db ..."
    isql <<< 'sparql select ?g (count(*) as ?c) where { graph ?g { ?s ?p ?o } } order by desc(?c) ;'
    echo "warmup db finished"
    date --rfc-3339=seconds

    # setup for triple insertion
    isql <<< 'grant SPARQL_UPDATE to "SPARQL" ;'

    # echo "db log"
    # cat virt*/db/virtuoso.log
}

if [[ -d "$TMP_DIR" ]] ; then
    echo "$TMP_DIR exists..."
    start_db
    exit 0
fi


echo "creating db dir: $TMP_DIR..."
mkdir -p "$TMP_DIR"
cd "$TMP_DIR"
echo -n "extracting db $pack... "
date --rfc-3339=seconds
#dd if="$pack" bs=2G | pv | "$PACKER" -d | tar -xvf -
#pv -tbre -i 10 -f "$pack" > >( dd bs=2G | "$PACKER" -d | tar -xvf - ) 2> >( tr '\r' '\n' >&2 )
#pv -i 10 -f "$pack" > >( dd bs=2G | "$PACKER" -d | tar -xvf - ) 2> >( tr '\r' '\n' >&2 )
#(pv -i 10 -f "$pack" | dd bs=2G | "$PACKER" -d | tar -xvf - ) 2>&1 | tr '\r' '\n'
if [[ "$PACKER" = "none" ]] ; then
    tar -xvf "$pack"
else
    if [[ "$PACKER" == *pixz ]] ; then
        $PACKER -k -d -t -i "$pack" | tar -xvf -
    else
        dd bs=2G if="$pack" | "$PACKER" -d | tar -xvf -
    fi
fi
echo -n "extraction complete "
date --rfc-3339=seconds
#cp "$ini" ./

start_db
