#!/usr/bin/env bash
# Cell width sweep, 8 cells of width 2R against 27 cells of width R.
# Run from Git Bash: tools/sweep_cellwidth.sh
# Appends everything to results/cell_width.csv
#
# Swept across N because the two effects pull in opposite directions and their
# balance depends on how crowded the cells are:
#   - width R searches 27 * R^3 of volume, width 2R searches 8 * (2R)^3 = 64 R^3
#   - but width R makes 42^3 = 74088 cells to clear each frame instead of 22^3

set -e
cd "$(dirname "$0")/../build"
EXE=./bin/Release/cis5650_boids.exe

rm -f benchmark_results.csv
start=$(date +%s)

for cw in 2 1; do
  for mode in scattered coherent; do
    for n in 5000 20000 100000 500000 1000000; do

      case $n in
        5000|20000)     warmup=300; frames=5000 ;;
        100000)         warmup=300; frames=1000 ;;
        *)              warmup=100; frames=300  ;;
      esac

      # total for the headline comparison, stages to see which part moved
      for bench in total stages; do
        for run in 1 2 3; do
          echo "cellwidth=${cw}R $mode N=$n $bench  (run $run)"
          $EXE --mode=$mode --n=$n --cellwidth=$cw --visualize=0 \
               --bench=$bench --warmup=$warmup --frames=$frames > /dev/null
        done
      done

    done
  done
done

mkdir -p ../results
mv benchmark_results.csv ../results/cell_width.csv
echo "done in $(( $(date +%s) - start ))s -> results/cell_width.csv"
