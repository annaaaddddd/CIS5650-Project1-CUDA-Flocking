#!/usr/bin/env bash
# N scaling sweep. Run from Git Bash: tools/sweep.sh
# Appends everything to results/n_scaling.csv

set -e
cd "$(dirname "$0")/../build"
EXE=./bin/Release/cis5650_boids.exe

rm -f benchmark_results.csv
start=$(date +%s)

for mode in naive scattered coherent; do
  for n in 1000 2000 5000 10000 20000 50000 100000 200000 500000 1000000; do

    # naive is O(N^2). Past 200k one run takes longer than the rest of the sweep.
    if [ "$mode" = naive ] && [ "$n" -gt 200000 ]; then continue; fi

    # Fewer frames at large N so no single run takes more than about a minute.
    case $n in
      1000|2000|5000|10000|20000) warmup=300; frames=5000 ;;
      50000|100000)               warmup=300; frames=1000 ;;
      *)                          warmup=100; frames=300  ;;
    esac

    for run in 1 2 3; do
      # Application FPS with and without visualization -- the assignment's plots
      for vis in 0 1; do
        echo "$mode N=$n vis=$vis fps  (run $run)"
        $EXE --mode=$mode --n=$n --visualize=$vis --bench=fps \
             --warmup=$warmup --frames=$frames > /dev/null
      done

      # GPU step time. Doesn't depend on the renderer, so visualization off only.
      echo "$mode N=$n total    (run $run)"
      $EXE --mode=$mode --n=$n --visualize=0 --bench=total \
           --warmup=$warmup --frames=$frames > /dev/null
    done

  done
done

mkdir -p ../results
mv benchmark_results.csv ../results/n_scaling.csv
echo "done in $(( $(date +%s) - start ))s -> results/n_scaling.csv"
