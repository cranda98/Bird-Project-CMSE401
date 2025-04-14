#!/bin/bash

INPUT="cmse2.txt"
BEST_FITNESS=999999
BEST_FILE="serial_best.txt"

rm -f $BEST_FILE

for SEED in {1..10}
do
    echo "Running seed $SEED..."
    
    OUTFILE="serial_output_seed_$SEED.txt"
    { time ./revROC $INPUT $SEED ; } &> $OUTFILE

    FITNESS=$(grep "Result Fitness=" $OUTFILE | awk -F'=' '{print $2}' | awk '{print $1}')
    
    echo "Seed $SEED → Fitness $FITNESS" >> timecheck_summary.txt

    if [ "$FITNESS" -lt "$BEST_FITNESS" ]; then
        BEST_FITNESS=$FITNESS
        cp $OUTFILE $BEST_FILE
    fi
done

echo "Best Fitness: $BEST_FITNESS" >> timecheck_summary.txt
