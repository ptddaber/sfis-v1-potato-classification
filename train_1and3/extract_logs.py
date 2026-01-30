import os
import re
import json
import csv

# ================= Configuration =================
# Directory containing log files
logs_dir = 'xx/logs_lau'

# Base directory for result output
target_base_dir = 'xx/Datasetforpt/train_1and3/outputs'

# Directory for parsed results
output_results_dir = os.path.join(target_base_dir, 'Parsed_Results')

# Ensure output directory exists
if not os.path.exists(output_results_dir):
    os.makedirs(output_results_dir)

# ================= Functions =================

def parse_log_filename(filename):
    """
    Parse filename, format assumed to be: [Model_Name]_[JobID].out
    Examples: 
      effnet_b0_2742389.out -> model: efficientnet_b0
      mobnet_v3_large_2742395.out -> model: mobilenet_v3_large
    """
    if not filename.endswith('.out'):
        return None
    
    # Remove suffix
    name_body = filename.replace('.out', '')
    
    # Split by underscore
    parts = name_body.split('_')
    
    # Simple validation: should have at least [model_name] and [ID] parts
    if len(parts) < 2:
        return None
        
    # Last part is usually Job ID (numeric)
    job_id = parts[-1]
    
    # All preceding parts form the model name
    # e.g., ['mobnet', 'v3', 'large', '2742395'] -> 'mobnet_v3_large'
    model_raw = "_".join(parts[:-1])
    
    # Model name standardization mapping (map abbreviations to standard names)
    model_map = {
        'convnext_t': 'convnext_tiny',
        'effnet_b0': 'efficientnet_b0',
        'effnet_v2s': 'efficientnet_v2_s',
        'mobnet_v2': 'mobilenet_v2',
        'mobnet_v3_large': 'mobilenet_v3_large',
        'resnet101': 'resnet101',
        'swin_t': 'swin_t'
    }
    
    # Use standard name if in mapping, otherwise use original name
    model_name = model_map.get(model_raw, model_raw)
    
    return {
        'model': model_name,
        'job_id': job_id
    }

def extract_metrics_from_log(filepath):
    """Extract various metrics from log file content"""
    metrics = {}
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
        
    # 1. Extract Test Accuracy
    # Format example: "Test Acc: 0.9901"
    acc_match = re.search(r'Test Acc: ([\d\.]+)', content)
    if acc_match:
        metrics['accuracy'] = float(acc_match.group(1))
        
    # 2. Extract Macro Metrics
    # Format example: "Precision (macro): 0.9891  Recall (macro): 0.9912  F1 (macro): 0.9901"
    macro_match = re.search(r'Precision \(macro\): ([\d\.]+) +Recall \(macro\): ([\d\.]+) +F1 \(macro\): ([\d\.]+)', content)
    if macro_match:
        metrics['precision_macro'] = float(macro_match.group(1))
        metrics['recall_macro'] = float(macro_match.group(2))
        metrics['f1_macro'] = float(macro_match.group(3))
        
    # 3. Extract Weighted Metrics
    weighted_match = re.search(r'Precision \(weighted\): ([\d\.]+) +Recall \(weighted\): ([\d\.]+) +F1 \(weighted\): ([\d\.]+)', content)
    if weighted_match:
        metrics['precision_weighted'] = float(weighted_match.group(1))
        metrics['recall_weighted'] = float(weighted_match.group(2))
        metrics['f1_weighted'] = float(weighted_match.group(3))
        
    # 4. Extract Inference Time
    # Format example: "... latency 0.0003s/img, 3389.15 img/s"
    infer_match = re.search(r'latency ([\d\.]+)s/img, ([\d\.]+) img/s', content)
    if infer_match:
        metrics['latency'] = float(infer_match.group(1))
        metrics['img_per_sec'] = float(infer_match.group(2))
        
    # 5. Extract Params and FLOPs
    params_match = re.search(r'Params: ([\d,]+)', content)
    if params_match:
        metrics['params'] = int(params_match.group(1).replace(',', ''))
        
    flops_match = re.search(r'FLOPs \(approx\): ([\d\.]+) GFLOPs', content)
    if flops_match:
        metrics['gflops'] = float(flops_match.group(1))
        
    return metrics

# ================= Main Program Logic =================

results = []
log_files = os.listdir(logs_dir)

# Print header
print(f"{'Model':<25} | {'Job ID':<10} | {'Acc':<8} | {'F1(M)':<8} | {'Filename'}")
print("-" * 90)

for filename in sorted(log_files):
    # 1. Parse filename
    info = parse_log_filename(filename)
    if not info:
        # Not a target .out file, skip
        continue
    
    model = info['model']
    job_id = info['job_id']
    
    # 2. Extract content metrics
    filepath = os.path.join(logs_dir, filename)
    metrics = extract_metrics_from_log(filepath)
    
    if not metrics:
        print(f"Skipping {filename} (No metrics found inside file)")
        continue
    
    # 3. Merge information
    metrics['model'] = model
    metrics['job_id'] = job_id
    metrics['source_log'] = filename
    
    results.append(metrics)
    
    # 4. Print progress
    acc_str = f"{metrics.get('accuracy', 0):.4f}"
    f1_str = f"{metrics.get('f1_macro', 0):.4f}"
    print(f"{model:<25} | {job_id:<10} | {acc_str:<8} | {f1_str:<8} | {filename}")
    
    # 5. Save individual JSON file
    # Filename format: summary_metrics_[model]_[job_id].json
    out_file = os.path.join(output_results_dir, f"summary_metrics_{model}_{job_id}.json")
    with open(out_file, 'w') as f:
        json.dump(metrics, f, indent=4)

# ================= Generate Summary CSV =================

if results:
    csv_file = os.path.join(target_base_dir, 'all_models_summary.csv')
    
    # Define CSV column order
    fieldnames = [
        'model', 'job_id', 
        'accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 
        'f1_weighted', 'precision_weighted', 'recall_weighted', 
        'params', 'gflops', 'latency', 'img_per_sec', 
        'source_log'
    ]

    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            # Filter out extra keys not in fieldnames to prevent errors; fill missing keys with empty string
            row = {k: res.get(k, '') for k in fieldnames}
            writer.writerow(row)

    print(f"\nSummary CSV saved to: {csv_file}")
    print(f"Individual JSONs saved to: {output_results_dir}")
else:
    print("\nNo valid log files processed.")
