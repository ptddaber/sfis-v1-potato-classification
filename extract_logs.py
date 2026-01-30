
import os
import re
import json
import shutil

# Define source and target directories
logs_dir = 'xx/logs_multi'
target_base_dir = 'xx/Datasetforpt/dataset_origin/duzixunlian/outputs'
dirs = {
    'pld': os.path.join(target_base_dir, 'PLD_Results'),
    'pv': os.path.join(target_base_dir, 'PV_Results'),
    'ue': os.path.join(target_base_dir, 'UE_Results'),
    'unknown': os.path.join(target_base_dir, 'Unclassified')
}

# Ensure target directories exist
for d in dirs.values():
    if not os.path.exists(d):
        os.makedirs(d)

# Define pattern to match log files
# e.g., convnext_t_pld_2742052.out
# parts: [model]_[dataset]_[jobid].out
# models usually have _ but dataset is clear: pld, pv, ue

def parse_log_filename(filename):
    if not filename.endswith('.out'):
        return None
    
    # Special case for potato_*.out files if needed, but let's focus on structured ones first
    if filename.startswith('potato_'):
        # e.g. potato_resnet101_2741916.out -> dataset unknown from filename, usually pld/pv?
        # user said "three datasets", potato_resnet* seemed to be earlier experiments.
        # We can check content for dataset name.
        return {'model': 'resnet_unknown', 'dataset': 'unknown', 'is_potato': True}
    
    # Standard pattern: [model_part]_[dataset]_[jobid].out
    # Datasets are pld, pv, ue.
    if '_pld_' in filename:
        ds = 'pld'
    elif '_pv_' in filename:
        ds = 'pv'
    elif '_ue_' in filename:
        ds = 'ue'
    else:
        return None # Skip unknown patterns
        
    parts = filename.split(f'_{ds}_')
    if len(parts) != 2:
        return None
        
    model_raw = parts[0]
    # Normalize model names
    model_map = {
        'convnext_t': 'convnext_tiny',
        'effnet_b0': 'efficientnet_b0',
        'effnet_v2s': 'efficientnet_v2_s',
        'mobnet_v2': 'mobilenet_v2',
        'mobnet_v3': 'mobilenet_v3_large', # default was large
        'resnet101': 'resnet101',
        'swin_t': 'swin_t'
    }
    
    model = model_map.get(model_raw, model_raw)
    return {'model': model, 'dataset': ds, 'is_potato': False}

def extract_metrics_from_log(filepath):
    metrics = {}
    with open(filepath, 'r') as f:
        content = f.read()
        
    # 1. Extract Test Accuracy
    # "Test Acc: 0.9901"
    acc_match = re.search(r'Test Acc: ([\d\.]+)', content)
    if acc_match:
        metrics['accuracy'] = float(acc_match.group(1))
        
    # 2. Extract Macro Metrics
    # "Precision (macro): 0.9891  Recall (macro): 0.9912  F1 (macro): 0.9901"
    macro_match = re.search(r'Precision \(macro\): ([\d\.]+) +Recall \(macro\): ([\d\.]+) +F1 \(macro\): ([\d\.]+)', content)
    if macro_match:
        metrics['precision_macro'] = float(macro_match.group(1))
        metrics['recall_macro'] = float(macro_match.group(2))
        metrics['f1_macro'] = float(macro_match.group(3))
        
    # 3. Extract Weighted Metrics
    # "Precision (weighted): 0.9902  Recall (weighted): 0.9901  F1 (weighted): 0.9901"
    weighted_match = re.search(r'Precision \(weighted\): ([\d\.]+) +Recall \(weighted\): ([\d\.]+) +F1 \(weighted\): ([\d\.]+)', content)
    if weighted_match:
        metrics['precision_weighted'] = float(weighted_match.group(1))
        metrics['recall_weighted'] = float(weighted_match.group(2))
        metrics['f1_weighted'] = float(weighted_match.group(3))
        
    # 4. Extract Inference Time
    # "Inference time: total 1.17s, net forward 0.12s, latency 0.0003s/img, 3389.15 img/s"
    infer_match = re.search(r'Inference time: .* latency ([\d\.]+)s/img, ([\d\.]+) img/s', content)
    if infer_match:
        metrics['latency'] = float(infer_match.group(1))
        metrics['img_per_sec'] = float(infer_match.group(2))
        
    # 5. Extract FLOPs and Params
    # "Params: 4,011,391 (15.30 MB)"
    # "FLOPs (approx): 0.77 GFLOPs @224"
    params_match = re.search(r'Params: ([\d,]+)', content)
    if params_match:
        metrics['params'] = int(params_match.group(1).replace(',', ''))
        
    flops_match = re.search(r'FLOPs \(approx\): ([\d\.]+) GFLOPs', content)
    if flops_match:
        metrics['gflops'] = float(flops_match.group(1))
        
    return metrics

# Process Logs
results = []
log_files = os.listdir(logs_dir)

print(f"{'Dataset':<10} | {'Model':<20} | {'Acc':<8} | {'F1(M)':<8} | {'Log File'}")
print("-" * 80)

for filename in sorted(log_files):
    if not filename.endswith('.out'):
        continue
        
    info = parse_log_filename(filename)
    if not info:
        continue
    
    ds = info['dataset']
    model = info['model']
    
    # Skip 'potato' ones if they are redundant or if we only want the structured ones
    if info.get('is_potato'):
        # Try to determine dataset from content? Usually potato_* were early runs.
        # Let's skip them to avoid duplication unless necessary.
        continue
        
    filepath = os.path.join(logs_dir, filename)
    metrics = extract_metrics_from_log(filepath)
    
    if not metrics:
        print(f"Skipping {filename} (No metrics found)")
        continue
    
    # Add meta info
    metrics['dataset'] = ds
    metrics['model'] = model
    metrics['source_log'] = filename
    
    results.append(metrics)
    
    acc_str = f"{metrics.get('accuracy', 0):.4f}"
    f1_str = f"{metrics.get('f1_macro', 0):.4f}"
    print(f"{ds:<10} | {model:<20} | {acc_str:<8} | {f1_str:<8} | {filename}")
    
    # Save extracted metrics to JSON file in target folder
    # Filename: summary_metrics_[dataset]_[model].json
    target_dir = dirs[ds]
    out_file = os.path.join(target_dir, f"summary_metrics_{ds}_{model}.json")
    
    with open(out_file, 'w') as f:
        json.dump(metrics, f, indent=4)

# Create a combined CSV summary
import csv
csv_file = os.path.join(target_base_dir, 'all_experiments_summary.csv')
fieldnames = ['dataset', 'model', 'accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 
              'f1_weighted', 'precision_weighted', 'recall_weighted', 
              'params', 'gflops', 'latency', 'img_per_sec', 'source_log']

with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for res in results:
        # Filter keys to match fieldnames
        row = {k: res.get(k, '') for k in fieldnames}
        writer.writerow(row)

print(f"\nSummary CSV saved to {csv_file}")
print("Individual metric JSONs saved to respective dataset folders.")
