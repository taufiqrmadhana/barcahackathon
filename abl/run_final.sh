cd /c/Users/neo/Koding/barcahackathon/.claude/worktrees/s7b
export MTAH_DATA_DIR=/c/Users/neo/Koding/barcahackathon/data
PY=/c/Users/neo/Koding/barcahackathon/.venv/Scripts/python.exe
for r in f1 f2; do rm -rf cache_$r; mkdir cache_$r; cp ../../../cache/BAR-ATH_tactical_atd.npz cache_$r/
BARCA_CACHE_DIR=$PWD/cache_$r $PY src_s2/run.py --params src_s2/params_s7b.json --name s7b_$r --out $PWD/out_s7b > out_s7b/$r.log 2>&1; done
