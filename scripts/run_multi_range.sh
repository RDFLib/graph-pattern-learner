#!/usr/bin/env bash

start=$1
shift
end=$1
shift

for i in $(seq $start $end) ; do
        BUNDLE_POST="_id_$i" SEQ_NUMBER="$i" $@
done
