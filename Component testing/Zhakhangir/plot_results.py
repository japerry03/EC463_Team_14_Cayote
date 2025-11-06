# plot_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
import os

print("Searching for result files...")
result_files = glob.glob('results_pub*_sub*.json')

if not result_files:
    print("Error: No 'results_pub*_sub*.json' files found.")
    print("Please run the 'run_qos_tests.sh' script first.")
    exit()

print(f"Found {len(result_files)} result files. Loading data...")

# Store summary and detailed data
summary_data = []
all_message_data = []

# Load all result files
for f in result_files:
    with open(f, 'r') as file:
        data = json.load(file)

        summary = {
            "pub_qos": data['pub_qos'],
            "sub_qos": data['sub_qos'],
            "qos_pair": f"P{data['pub_qos']}/S{data['sub_qos']}",
            "loss_percentage": data['loss_percentage']
        }
        summary_data.append(summary)

        for msg in data['results']:
            msg_data = msg.copy()
            msg_data['pub_qos'] = data['pub_qos']
            msg_data['sub_qos'] = data['sub_qos']
            msg_data['qos_pair'] = f"P{data['pub_qos']}/S{data['sub_qos']}"
            all_message_data.append(msg_data)

df_summary = pd.DataFrame(summary_data).sort_values(by=['pub_qos', 'sub_qos'])
df_messages = pd.DataFrame(all_message_data)

print("Data loaded. Generating plots...")

# --- Plot 1: Packet Loss Percentage ---
plt.figure(figsize=(15, 7))
sns.barplot(
    data=df_summary,
    x='qos_pair',
    y='loss_percentage',
    hue='pub_qos',
    dodge=False
)
plt.title('MQTT Packet Loss by QoS Pair', fontsize=16)
plt.xlabel('QoS Pair (Publish/Subscribe)', fontsize=12)
plt.ylabel('Packet Loss (%)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('plot_1_packet_loss.jpg', format='jpeg', dpi=150, pil_kwargs={'quality': 95})
print("Saved plot_1_packet_loss.jpg")

# --- Plot 2: Latency Distribution (Box Plot) ---
plt.figure(figsize=(15, 7))
sns.boxplot(
    data=df_messages,
    x='qos_pair',
    y='latency_ms',
    order=df_summary['qos_pair']
)
plt.title('Latency Distribution by QoS Pair', fontsize=16)
plt.xlabel('QoS Pair (Publish/Subscribe)', fontsize=12)
plt.ylabel('Latency (ms)', fontsize=12)
plt.yscale('log')  # Log scale is helpful for latency
plt.grid(axis='y', linestyle='--', alpha=0.7)
# --- THIS LINE IS CORRECTED ---
plt.savefig('plot_2_latency_distribution.jpg', format='jpeg', dpi=150, pil_kwargs={'quality': 95})
print("Saved plot_2_latency_distribution.jpg")

# --- Plot 3: Transmission Speed Distribution (Box Plot) ---
df_messages['speed_KBps'] = df_messages['transmission_speed_Bps'] / 1024

plt.figure(figsize=(15, 7))
sns.boxplot(
    data=df_messages,
    x='qos_pair',
    y='speed_KBps',
    order=df_summary['qos_pair']
)
plt.title('Transmission Speed Distribution by QoS Pair', fontsize=16)
plt.xlabel('QoS Pair (Publish/Subscribe)', fontsize=12)
plt.ylabel('Transmission Speed (KB/s)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
# --- THIS LINE IS CORRECTED ---
plt.savefig('plot_3_speed_distribution.jpg', format='jpeg', dpi=150, pil_kwargs={'quality': 95})
print("Saved plot_3_speed_distribution.jpg")

print("\nAll plots generated successfully.")