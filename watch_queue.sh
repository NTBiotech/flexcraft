#! /usr/bin/bash

id=$(squeue | awk -F " " '{print $1}' | tail -n 1)

cat logs/adapt_${1}_${id}.${2} | tail -n 20
