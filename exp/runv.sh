#!/bin/bash
# usage: exp/runv.sh name [key=val ...]   (frozen shape = s7b run's, for fast experiments)
cd /c/Users/neo/Koding/barcahackathon/.claude/worktrees/s8
export MTAH_DATA_DIR=C:/Users/neo/Koding/barcahackathon/data BARCA_CACHE_DIR=$PWD/cache BARCA_SHAPE_FROZEN=$PWD/cache/s2shape_b524176329ef03ad.npy
N=$1; shift
../../../.venv/Scripts/python.exe src_s2/run.py --params ${PARAMS:-src_s2/params_s7b.json} --name $N --out $PWD/out_exp --set "$@" | tail -2
