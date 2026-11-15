cd /c/Users/neo/Koding/barcahackathon/.claude/worktrees/s7b
export MTAH_DATA_DIR=/c/Users/neo/Koding/barcahackathon/data
unset BARCA_SHAPE_FROZEN
/c/Users/neo/Koding/barcahackathon/.venv/Scripts/python.exe /c/Users/neo/Koding/barcahackathon/personal/loop/ablate.py --workdir $PWD --base src_s2/params_s4.json --variants abl/variants_graded.json --cache-env BARCA_CACHE_DIR=$PWD/cache_abl > abl/table_graded.md 2> abl/table_graded.err
