import re
from collections import defaultdict
import subprocess
import os
import csv
import shutil
from datetime import datetime
from tkinter import Tk, filedialog

# 📂 File picker dialog
root = Tk()
root.withdraw()
log_file = filedialog.askopenfilename(title="Choose log file to analyze", filetypes=[("Text files", "*.txt"), ("Log files", "*.log")], initialdir="/tmp")
if not log_file:
    print("No file selected.")
    exit(1)

# 🧹 Backup existing output
out_path = "/tmp/analz.txt"
if os.path.exists(out_path):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"/tmp/analz.bak.{timestamp}.txt"
    shutil.move(out_path, backup_path)
    print(f"Backed up old file to {backup_path}")

# 🧵 Run grep pipeline via shell
cmd = f'grep -E "DoApplyAction|Initializing game" "{log_file}" | grep -v "completed" > {out_path}'
subprocess.run(cmd, shell=True, check=True)

print(f"Filtered output saved to {out_path}")
# Optional: proceed to parse with your existing Python logic


pattern = re.compile(r'DoApplyAction: Player (\w+) applying move type (\d+) from action (\d+)')
counts = defaultdict(lambda: defaultdict(int))

with open(out_path) as f:
    for line in f:
        match = pattern.search(line)
        if match:
            player, move_type, action = match.groups()
            counts[player][move_type] += 1

# Pretty print results
for player, moves in counts.items():
    print(f"Player: {player}")
    for move, count in sorted(moves.items()):
        print(f"  Move type {move}: {count}")

# === send the output to a CSV file
log_file = out_path
output_csv = "/tmp/analz_output.csv"

init_pattern = re.compile(r'Initializing game')
move_pattern = re.compile(r'(\d{2}:\d{2}:\d{2}).*Player (\w+) applying move type (\d+) from action (\d+)')

game_id = 0
rows = []

with open(log_file) as f:
    for line in f:
        if init_pattern.search(line):
            game_id += 1
        match = move_pattern.search(line)
        if match:
            timestamp, player, move_type, action_id = match.groups()
            rows.append([game_id, timestamp, player, move_type, action_id])

# 💾 Save to CSV
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['game_id', 'timestamp', 'player', 'move_type', 'action_id'])
    writer.writerows(rows)

print(f"Formatted CSV saved to {output_csv}")
