#! /usr/bin/bash
for f in data/adapt/full_run_seed/*; do
echo $f
parent="$(dirname "$f")"
parent="$(basename "$parent")"
base="$(basename "$f")"
subdir="$parent-$base"
echo $subdir
mkdir ./config_rmsds/$subdir
cp $f/rmsd_new.csv ./config_rmsds/$subdir/
rmdir ./config_rmsds/$subdir
done