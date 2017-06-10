#!/usr/bin/env bash

number=$1
shift

for i in $(seq 1 $number) ; do
        BUNDLE_POST="_multi_$i" $@
done
