#!/bin/bash
# run_demo.sh
# A short, fast demo script for the QoS 0/0 test.

# --- Configuration for a fast demo ---
NUM_MESSAGES=100       # Send 50 messages
INTERVAL=0.1          # Send one message every 0.1 seconds (Total publish time: 5 sec)
TEST_DURATION=15      # Subscriber listens for 15 seconds (plenty of buffer)
PUB_QOS=0
SUB_QOS=0

echo "Starting short class demo: QoS 0/0 Test..."
echo "Will send $NUM_MESSAGES messages at $INTERVAL sec intervals."
echo ""

# --- 1. Clean up old results ---
echo " Cleaning up old result files..."
rm -f results_pub*_sub*.json
rm -f plot_*.jpg

# --- 2. Run the Single Test Pair ---
echo "--------------------------------------------------------"
echo "--- Starting Test: Pub QoS $PUB_QOS / Sub QoS $SUB_QOS ---"
echo "--------------------------------------------------------"

# Start the subscriber in the background
echo "Starting subscriber (background)..."
python3 latency_tester.py --pub_qos $PUB_QOS --sub_qos $SUB_QOS --expected_num $NUM_MESSAGES --duration $TEST_DURATION &

# Capture the Process ID (PID) of the subscriber
sub_pid=$!
echo "Subscriber PID: $sub_pid"

# Give the subscriber a moment to connect
sleep 2

# Start the publisher in the foreground
echo "Starting publisher (foreground)..."
python3 robots.py --qos $PUB_QOS --num_messages $NUM_MESSAGES --interval $INTERVAL

echo "Publisher finished."

# Wait for the subscriber to finish its 15-second run
echo "Waiting for subscriber (PID $sub_pid) to finish..."
wait $sub_pid

echo "Subscriber finished. Test complete."
echo ""

# --- 3. Generate Plots ---
echo "Generating plots..."
python3 plot_results.py

echo "--------------------------------------------------------"
echo " All done! Check your directory for 'plot_*.jpg' files."
echo "--------------------------------------------------------"