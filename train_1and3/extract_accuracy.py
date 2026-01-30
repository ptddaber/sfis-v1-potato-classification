import os
import glob
import json

OUTPUT_DIR = "xx/Datasetforpt/train_1and3/evaluate_output"

# Read all JSON files
json_pattern = os.path.join(OUTPUT_DIR, "test_metrics_*.json")
json_files = glob.glob(json_pattern)

# Used to store all accuracy data for each model
model_accuracies = {}

for jf in json_files:
    try:
        with open(jf, 'r') as f:
            d = json.load(f)
            if 'model_name' in d and 'accuracy' in d:
                model_name = d['model_name']
                accuracy = d['accuracy']
                
                if model_name not in model_accuracies:
                    model_accuracies[model_name] = []
                model_accuracies[model_name].append(accuracy)
    except Exception as e:
        print(f"Error reading {jf}: {e}")

# Print average accuracy for each model
print("=" * 60)
print("Model Accuracy Statistics")
print("=" * 60)
print(f"{'Model Name':<30} {'Average Accuracy':<15} {'Max Accuracy':<15}")
print("-" * 60)

# Sort by average accuracy
sorted_models = sorted(model_accuracies.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True)

for model_name, accuracies in sorted_models:
    avg_acc = sum(accuracies) / len(accuracies) * 100
    max_acc = max(accuracies) * 100
    print(f"{model_name:<30} {avg_acc:.2f}%         {max_acc:.2f}%")

print("=" * 60)
print(f"Total {len(sorted_models)} models")
