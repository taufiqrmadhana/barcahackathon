cd /c/Users/neo/Koding/barcahackathon/.claude/worktrees/s8
export MTAH_DATA_DIR=C:/Users/neo/Koding/barcahackathon/data
unset BARCA_SHAPE_FROZEN
PY=../../../.venv/Scripts/python.exe
mkdir -p out_final
for r in f1 f2; do rm -rf cache_$r; mkdir cache_$r; cp ../../../cache/BAR-ATH_tactical_atd.npz cache_$r/
BARCA_CACHE_DIR=$PWD/cache_$r $PY src_s2/run.py --params src_s2/params_s8.json --name s8_$r --out $PWD/out_final > out_final/$r.log 2>&1; done
sha256sum out_final/s8_f1.csv out_final/s8_f2.csv > out_final/sha.txt
