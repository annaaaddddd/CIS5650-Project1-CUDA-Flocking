#!/usr/bin/env bash
# Block size sweep. Run from Git Bash: tools/sweep_block.sh
# Appends everything to results/block_size.csv
#
# Two boid counts on purpose: 5000 sits below the point where the GPU
# saturates, 100000 sits above it. If performance at 5000 is limited by not
# having enough parallel work, block size should barely matter there, because
# the total warp count is ceil(5000/32) = 157 whatever the block size is.

set -e
cd "$(dirname "$0")/../build"
EXE=./bin/Release/cis5650_boids.exe

rm -f benchmark_results.csv
start=$(date +%s)

for mode in naive scattered coherent; do
  for n in 5000 100000; do

    case $n in
      5000)   warmup=300; frames=5000 ;;
      *)      warmup=300; frames=1000 ;;
    esac

    for block in 32 64 128 256 512 1024; do
      for run in 1 2 3; do
        echo "$mode N=$n block=$block  (run $run)"
        $EXE --mode=$mode --n=$n --block=$block --visualize=0 --bench=fps \
             --warmup=$warmup --frames=$frames > /dev/null
      done
    done

  done
done

mkdir -p ../results
mv benchmark_results.csv ../results/block_size.csv
echo "done in $(( $(date +%s) - start ))s -> results/block_size.csv"
