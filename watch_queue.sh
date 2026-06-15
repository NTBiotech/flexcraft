#! /usr/bin/bash

id=$(squeue | grep toulouse | awk -F " " '{print $1}')

cat logs/adapt_${id}.${1} | tail -n 20
