#!/usr/bin/env bash
cd "$(dirname $0)"
rm -r logs/*.log* logs/*memory*.png logs/error_logs_*/ || exit 0
