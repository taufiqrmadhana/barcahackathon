cd /c/Users/neo/Koding/barcahackathon/.claude/worktrees/s8
export MTAH_DATA_DIR=C:/Users/neo/Koding/barcahackathon/data BARCA_CACHE_DIR=$PWD/cache BARCA_SHAPE_FROZEN=$PWD/cache/s2shape_b524176329ef03ad.npy
PY=../../../.venv/Scripts/python.exe
P=src_s2/params_s8pre.json
echo "base: $($PY src_s2/hidden.py --params $P --quiet)"
for kv in use_teleport=0 use_interp=0 use_merge=0 use_still_missing_speed=0 use_shape_gap=0 use_min_spell=0 use_shape_restart=0 use_restart_lag=0 "use_teleport=0 use_interp=0 use_merge=0 use_still_missing_speed=0 use_shape_gap=0 use_min_spell=0 use_shape_restart=0 use_restart_lag=0" use_gap_kick=0; do
 echo "$kv: $($PY src_s2/hidden.py --params $P --quiet --set $kv)"
done
