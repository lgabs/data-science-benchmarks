import argparse

import random
import time
import matplotlib.pyplot as plt
import pandas as pd

from utils import create_folder_if_not_exists

# Initialize argparse
parser = argparse.ArgumentParser(description='Benchmark various data science tasks.')
parser.add_argument('-n', '--number', type=int, default=100_000_000, help='Number of integers in the list.')
args = parser.parse_args()
n = args.number

# Initialize variables
task_names = ["Creation", "Square", "Square Root", "Multiply", "Divide", "Int Divide"]
task_times = []

SAVE_DIR = "cpu_and_memory_tasks/pure_python/results"
create_folder_if_not_exists(SAVE_DIR)

# Generate the list of random integers
start_time = time.time()
l = [random.randint(100, 999) for _ in range(n)]
end_time = time.time()
task_times.append(end_time - start_time)

# Perform the tasks and measure time for each
# 1. Square each item
start_time = time.time()
squares = [x ** 2 for x in l]
end_time = time.time()
task_times.append(end_time - start_time)

# 2. Take square root of each item
start_time = time.time()
square_roots = [x ** 0.5 for x in l]
end_time = time.time()
task_times.append(end_time - start_time)

# 3. Multiply corresponding squares and square roots
start_time = time.time()
multiplied = [s * r for s, r in zip(squares, square_roots)]
end_time = time.time()
task_times.append(end_time - start_time)

# 4. Divide corresponding squares and square roots
start_time = time.time()
divided = [s / r for s, r in zip(squares, square_roots)]
end_time = time.time()
task_times.append(end_time - start_time)

# 5. Perform integer division of corresponding squares and square roots
start_time = time.time()
int_divided = [s // r for s, r in zip(squares, square_roots)]
end_time = time.time()
task_times.append(end_time - start_time)

# Calculate total time
total_time = sum(task_times)

# Create a single plot with all task times and total time
fig, ax = plt.subplots(figsize=(10, 6))
all_task_names = task_names + ["Total"]
all_task_times = task_times + [total_time]
ax.bar(all_task_names, all_task_times, color=['blue']*len(task_names) + ['red'])
ax.set_title(f"Benchmark Time for Each Task and Total Time\n({n} samples)")
ax.set_xlabel("Tasks")
ax.set_ylabel("Time (seconds)")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/benchmark_pure_python_{n}_samples_results.png")

# Save to CSV
df = pd.DataFrame({
    "Task": all_task_names,
    "Time (seconds)": all_task_times
})
df.to_csv(f"{SAVE_DIR}/benchmark_pure_python_{n}_samples_results.csv", index=False)