import argparse

import lightgbm as lgb
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SAVE_DIR = "cpu_and_memory_tasks/lightgbm/results"
from utils import create_folder_if_not_exists
create_folder_if_not_exists(SAVE_DIR)

# Initialize argparse
parser = argparse.ArgumentParser(description='Benchmark various data science tasks.')
parser.add_argument('-n', '--number', type=int, default=10_000_000, help='Number of integers in the list.')
args = parser.parse_args()
n = args.number

# Initialize variables
features = 50  # Number of features

# Generate synthetic data
print("Generating dataset...")
X = np.random.rand(n, features)
y = np.random.randint(2, size=n)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize LightGBM parameters
params = {
    "objective": "binary",
    "metric": "binary_error",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "early_stopping_rounds": 50,
}

# Create dataset for LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# Initialize task names and times
task_names = ["Data Generation", "Training", "Prediction"]
task_times = []

# Time taken for data generation (already generated)
task_times.append(0)  # Placeholder for demonstration

# Train the model and record time
print("Training model...")
start_time = time.time()
clf = lgb.train(params, train_data, valid_sets=[test_data])
end_time = time.time()
task_times.append(end_time - start_time)

# Make predictions and record time
print("Making predictions...")
start_time = time.time()
y_pred = clf.predict(X_test, num_iteration=clf.best_iteration)
y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]
end_time = time.time()
task_times.append(end_time - start_time)

# Calculate total time
total_time = sum(task_times)

print("Saving files...")
# Create a single plot with all task times and total time
fig, ax = plt.subplots(figsize=(10, 6))
all_task_names = task_names + ["Total"]
all_task_times = task_times + [total_time]
ax.bar(all_task_names, all_task_times, color=['blue']*len(task_names) + ['red'])
ax.set_title(f"Benchmark Time for Each Task and Total Time\n({n} samples)")
ax.set_xlabel("Tasks")
ax.set_ylabel("Time (seconds)")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/lightgbm_benchmark_{n}_samples_results.png")

# Save to CSV
df = pd.DataFrame({
    "Task": all_task_names,
    "Time (seconds)": all_task_times
})
df.to_csv(f"{SAVE_DIR}/lightgbm_benchmark_{n}_samples_results.csv", index=False)
