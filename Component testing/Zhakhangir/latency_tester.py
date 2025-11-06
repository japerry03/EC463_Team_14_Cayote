# backend/api/src/routes/latency_tester.py
import paho.mqtt.client as mqtt
import json
import time
import os
import argparse  # Import argparse
from dotenv import load_dotenv
from urllib.parse import urlparse
import statistics

# --- Configuration ---
load_dotenv()
MQTT_BROKER_URL = os.getenv('MQTT_BROKER_URL')
USERNAME = os.getenv('MQTT_USERNAME')
PASSWORD = os.getenv('MQTT_PASSWORD')
TOPIC = "robot/cayote001/telemetry"

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="MQTT Latency Tester")
parser.add_argument('--sub_qos', type=int, choices=[0, 1, 2], default=0, help='Subscribe QoS level')
parser.add_argument('--pub_qos', type=int, choices=[0, 1, 2], default=0, help='Expected Publish QoS (for filename)')
parser.add_argument('--expected_num', type=int, default=100, help='Number of messages expected')
parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds')
args = parser.parse_args()

# This list will store dictionaries of results
results_data = []
output_filename = f"results_pub{args.pub_qos}_sub{args.sub_qos}.json"

if not MQTT_BROKER_URL:
    raise ValueError("MQTT_BROKER_URL environment variable not set.")

if not MQTT_BROKER_URL.startswith(('ws://', 'wss://')):
    full_url = 'wss://' + MQTT_BROKER_URL
else:
    full_url = MQTT_BROKER_URL

parsed_url = urlparse(full_url)
BROKER_HOST = parsed_url.hostname
BROKER_PORT = parsed_url.port or 8884


# --- MQTT Callbacks ---
def on_connect(client, userdata, flags, rc, properties):
    if rc == 0:
        print("Success: Connected to HiveMQ Cloud!")
        # Subscribe with the specified QoS
        client.subscribe(TOPIC, qos=args.sub_qos)
        print(f"Subscribed to topic '{TOPIC}' with QoS {args.sub_qos}.")
    else:
        print(f"Failed to connect, return code {rc}\n")


def on_message(client, userdata, msg):
    """Callback for when a message is received."""
    receive_time = time.time()
    try:
        payload = json.loads(msg.payload)
        sent_time = payload.get("timestamp")
        message_id = payload.get("message_id")
        payload_size_bytes = len(msg.payload)

        if sent_time and message_id is not None:
            latency_sec = receive_time - sent_time
            latency_ms = latency_sec * 1000

            # Calculate speed: Bytes / second
            # Avoid division by zero if latency is extremely low
            transmission_speed_Bps = payload_size_bytes / latency_sec if latency_sec > 0 else 0

            # Store all relevant data
            results_data.append({
                "message_id": message_id,
                "latency_ms": latency_ms,
                "payload_size_bytes": payload_size_bytes,
                "transmission_speed_Bps": transmission_speed_Bps
            })
            print(
                f"Received Msg {message_id}. Latency: {latency_ms:.2f} ms. Speed: {transmission_speed_Bps / 1024:.2f} KB/s")
        else:
            print("Received message without 'timestamp' or 'message_id'.")

    except json.JSONDecodeError:
        print("Could not decode message payload.")


# --- Client Setup and Main Loop ---
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, transport="websockets")
client.username_pw_set(USERNAME, PASSWORD)
client.tls_set()
client.on_connect = on_connect
client.on_message = on_message

try:
    print(f"Connecting to broker at {BROKER_HOST}:{BROKER_PORT}...")
    client.connect(BROKER_HOST, BROKER_PORT)
except Exception as e:
    print(f"Error: Could not connect to broker. {e}")
    exit()

print(f"Starting latency test for {args.duration} seconds...")
print(f"Will save results to {output_filename}")
client.loop_start()
time.sleep(args.duration)  # Run the test for the specified duration
client.loop_stop()
client.disconnect()
print("Test finished. Disconnected from MQTT Broker.")

# --- Process Results ---
if results_data:
    latencies = [r['latency_ms'] for r in results_data]
    speeds = [r['transmission_speed_Bps'] for r in results_data]
    received_ids = {r['message_id'] for r in results_data}
    expected_ids = set(range(args.expected_num))

    # Calculate packet loss
    lost_ids = expected_ids.difference(received_ids)
    messages_received = len(received_ids)
    messages_lost = len(lost_ids)
    loss_percentage = (messages_lost / args.expected_num) * 100

    print("\n--- Test Run Summary ---")
    print(f"QoS (Pub/Sub):   {args.pub_qos} / {args.sub_qos}")
    print(f"Messages Expected: {args.expected_num}")
    print(f"Messages Received: {messages_received}")
    print(f"Messages Lost:     {messages_lost} ({loss_percentage:.2f}%)")
    print("\n--- Performance Metrics ---")
    print(f"Average Latency: {statistics.mean(latencies):.2f} ms")
    print(f"Median Latency:  {statistics.median(latencies):.2f} ms")
    print(f"Min Latency:     {min(latencies):.2f} ms")
    print(f"Max Latency:     {max(latencies):.2f} ms")
    print(f"Avg. Speed:      {(statistics.mean(speeds) / 1024):.2f} KB/s")

    # Save raw data for plotting
    summary = {
        "pub_qos": args.pub_qos,
        "sub_qos": args.sub_qos,
        "expected_messages": args.expected_num,
        "received_messages": messages_received,
        "lost_messages": messages_lost,
        "loss_percentage": loss_percentage,
        "results": results_data
    }
    with open(output_filename, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"\nSuccessfully saved detailed results to {output_filename}")

else:
    print("\nNo messages were received during the test.")
    print(f"Messages Expected: {args.expected_num}")
    print(f"Messages Received: 0")
    print(f"Messages Lost:     {args.expected_num} (100.00%)")