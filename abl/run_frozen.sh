cd /c/Users/neo/Koding/barcahackathon/.claude/worktrees/s7b
export MTAH_DATA_DIR=/c/Users/neo/Koding/barcahackathon/data
export BARCA_SHAPE_FROZEN=$PWD/cache_r3/s2shape_3052903e5baf7d01.npy
/c/Users/neo/Koding/barcahackathon/.venv/Scripts/python.exe /c/Users/neo/Koding/barcahackathon/personal/loop/ablate.py --workdir $PWD --base src_s2/params_s4.json --variants abl/variants_frozen.json --cache-env BARCA_CACHE_DIR=$PWD/cache_abl > abl/table_frozen.md 2> abl/table_frozen.err
