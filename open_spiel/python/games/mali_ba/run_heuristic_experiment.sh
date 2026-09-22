#!/bin/bash
# run_heuristic_experiment.sh
# Runs 100 heuristic-only episodes, each with randomised kMancalaStep params,
# and records the win counts alongside the param values used.
#
# Usage: cd /media/robp/UD/Projects/open_spiel/open_spiel/python/games/mali_ba
#        bash run_heuristic_experiment.sh [num_iterations]
#
# Output: /tmp/heuristic_experiment_results.csv

SCRIPT_DIR="/media/robp/UD/Projects/open_spiel/open_spiel/python/games/mali_ba"
RESULTS_FILE="/tmp/heuristic_experiment_results.csv"
PARAMS_FILE="/tmp/mali_ba_heuristic_params.txt"
NUM_ITERATIONS=${1:-100}

cd "$SCRIPT_DIR" || { echo "ERROR: cannot cd to $SCRIPT_DIR"; exit 1; }

# Write CSV header
echo "iteration,mult_add_in,add_add_in,wins_rare_region,wins_timbuktu_coast,total_wins" > "$RESULTS_FILE"

echo "Starting $NUM_ITERATIONS iterations. Results -> $RESULTS_FILE"

for i in $(seq 1 "$NUM_ITERATIONS"); do
    echo ""
    echo "=== Iteration $i / $NUM_ITERATIONS ==="

    # Clean up log files from previous run
    rm -f /tmp/mali_ba*.log /tmp/test_region_fix.log /tmp/diag*.log /tmp/temp.log

    # Run training (randomised params are written by the script before actors start)
    python train_mali_ba.py \
        --heuristic_only \
        --randomise_heuristic_params \
        --num_actors 2 \
        --bootstrap_episodes 100 \
        --num_episodes 100 \
        --config_file mali_ba.ini \
        > /tmp/test_region_fix.log 2>&1

    # Read the params that were used this run
    if [ -f "$PARAMS_FILE" ]; then
        read MULT ADD < "$PARAMS_FILE"
    else
        MULT="0.0"
        ADD="0.0"
    fi

    # Count wins by condition
    WINS_RARE=$(grep "HEURISTIC_DIAG" /tmp/mali_ba.*.log 2>/dev/null \
        | grep "Game ended:" \
        | grep -o "reason='[^']*'" \
        | grep -c "Rare good from N regions")

    WINS_TIMB=$(grep "HEURISTIC_DIAG" /tmp/mali_ba.*.log 2>/dev/null \
        | grep "Game ended:" \
        | grep -o "reason='[^']*'" \
        | grep -c "Timbuktu to coast")

    TOTAL_WINS=$(( WINS_RARE + WINS_TIMB ))

    echo "  mult_add_in=$MULT  add_add_in=$ADD  rare_region=$WINS_RARE  timbuktu=$WINS_TIMB  total=$TOTAL_WINS"

    # Append to CSV
    echo "$i,$MULT,$ADD,$WINS_RARE,$WINS_TIMB,$TOTAL_WINS" >> "$RESULTS_FILE"
done

echo ""
echo "=== Experiment complete ==="
echo "Results saved to $RESULTS_FILE"
echo ""
echo "Top 10 runs by total wins:"
echo "iteration,mult_add_in,add_add_in,wins_rare_region,wins_timbuktu_coast,total_wins"
tail -n +2 "$RESULTS_FILE" | sort -t',' -k6 -rn | head -10
echo ""
echo "Summary stats (total_wins column):"
tail -n +2 "$RESULTS_FILE" | awk -F',' '
    BEGIN { sum=0; min=9999; max=0; n=0 }
    { sum+=$6; n++; if($6<min) min=$6; if($6>max) max=$6 }
    END { printf "  n=%d  mean=%.1f  min=%d  max=%d\n", n, sum/n, min, max }
'