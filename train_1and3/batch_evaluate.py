import os
import glob
import json
import torch
import torch.nn as nn
from torchvision import models
import matplotlib
matplotlib.use("Agg") # Draw in background, do not show window
import matplotlib.pyplot as plt
import numpy as np

# Import custom modules
from data_utils import create_dataloaders
from trainer import evaluate_on_test

# ================= Configuration =================
# Directory where checkpoint files are located
CHECKPOINT_DIR = "./checkpoints"

# Path for dataset split files (train.csv, val.csv, test.csv)
SPLITS_DIR = "xx/Datasetforpt/train_1and3/splits"

# Directory to store output results (JSON, PNG)
OUTPUT_DIR = "xx/Datasetforpt/train_1and3/evaluate_output"

BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= Model Architecture Mapping =================
MODEL_BUILDERS = {
    'resnet101': models.resnet101,
    'mobilenet_v2': models.mobilenet_v2,
    'mobilenet_v3_large': models.mobilenet_v3_large,
    'efficientnet_b0': models.efficientnet_b0,
    'efficientnet_v2_s': models.efficientnet_v2_s,
    'convnext_tiny': models.convnext_tiny,
    'swin_t': models.swin_t,
}

def load_model_structure(model_name_key, num_classes):
    """Build model backbone and modify classifier head based on name"""
    builder = MODEL_BUILDERS.get(model_name_key)
    if not builder:
        raise ValueError(f"Unknown model architecture for key: {model_name_key}")
    
    model = builder(pretrained=False)
    
    if model_name_key.startswith('resnet'):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name_key.startswith('mobilenet_v2'):
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name_key.startswith('mobilenet_v3'):
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name_key.startswith('efficientnet'):
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name_key.startswith('convnext'):
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif model_name_key.startswith('swin'):
        model.head = nn.Linear(model.head.in_features, num_classes)
    else:
        raise NotImplementedError(f"Head replacement not implemented for {model_name_key}")
        
    return model

def parse_filename(filename):
    """Parse filename to extract model architecture name"""
    filename = filename.lower()
    for key in MODEL_BUILDERS.keys():
        if filename.startswith(key):
            return key
    return None

def summarize_and_plot(output_dir):
    """Read JSON metrics and plot: 1. Inference speed 2. Bubble chart (Acc vs Size)"""
    print(f"Generating summary plots in {output_dir}...")
    
    json_pattern = os.path.join(output_dir, "test_metrics_*.json")
    json_files = glob.glob(json_pattern)
    
    data = []
    for jf in json_files:
        # Skip summary file itself and temporary files
        if "last_test_metrics" in os.path.basename(jf): continue
        try:
            with open(jf, 'r') as f:
                d = json.load(f)
                if 'model_name' in d and 'accuracy' in d:
                    data.append(d)
        except Exception as e:
            print(f"Error reading {jf}: {e}")

    if not data:
        print("[Warning] No valid metrics found to plot.")
        return

    # Extract and unique model names
    models_set = sorted(list(set([d['model_name'] for d in data])))
    # Assign colors
    cmap = plt.get_cmap("tab10")
    colors = {m: cmap(i % 10) for i, m in enumerate(models_set)}

    # ==========================================
    # Chart 1: Average Inference Speed (Horizontal Bar)
    # ==========================================
    model_speeds = {}
    for m in models_set:
        recs = [d for d in data if d['model_name'] == m]
        if recs:
            # Calculate average IPS
            avg_ips = sum([r['images_per_second'] for r in recs]) / len(recs)
            model_speeds[m] = avg_ips
    
    # Sort
    sorted_models_speed = sorted(model_speeds.items(), key=lambda x: x[1])
    names_s = [x[0] for x in sorted_models_speed]
    speeds_s = [x[1] for x in sorted_models_speed]
    colors_s = [colors[n] for n in names_s]

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    bars = ax1.barh(names_s, speeds_s, color=colors_s, alpha=0.7)
    ax1.set_xlabel('Speed (img/s)')
    ax1.set_title('Average Inference Speed (Images/sec)')
    ax1.grid(True, axis='x', linestyle='--', alpha=0.5)
    for bar in bars:
        width = bar.get_width()
        ax1.text(width + 50, bar.get_y() + bar.get_height()/2, f'{int(width)}', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_inference_speed.png"), dpi=300)
    plt.close(fig1)

    # ==========================================
    # Chart 2: Accuracy Bar Chart
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    model_data = []
    for model in models_set:
        recs = [d for d in data if d['model_name'] == model]
        if not recs: continue
        
        # Calculate average accuracy
        avg_acc = sum([r['accuracy'] for r in recs]) / len(recs) * 100
        params_million = recs[0]['params'] / 1e6
        flops_g = recs[0]['approx_flops'] / 1e9
        model_data.append({'name': model, 'acc': avg_acc, 'params': params_million, 'flops': flops_g})
    
    # Sort by accuracy
    model_data.sort(key=lambda x: x['acc'], reverse=True)
    x_pos = np.arange(len(model_data))
    model_names = [m['name'] for m in model_data]
    accuracies = [m['acc'] for m in model_data]
    bar_colors = [colors[m['name']] for m in model_data]
    
    # Plot bar chart (Accuracy)
    bars = ax2.bar(x_pos, accuracies, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Annotate accuracy values above bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{acc:.2f}%', 
                ha='center', va='bottom', fontsize=9)
    
    # Set x-axis labels
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Model Accuracy Comparison')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax2.set_ylim([min(accuracies) - 2, max(accuracies) + 3])
    
    # Add legend
    legend_elements = []
    for model, color in zip(model_names, bar_colors):
        legend_elements.append(plt.Rectangle((0,0), 1, 1, color=color, alpha=0.7, label=model))
    ax2.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), title="Models")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_accuracy_bar.png"), dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # ==========================================
    # Chart 3: Accuracy vs Model Size (Bubble Chart)
    # ==========================================
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    
    for model in models_set:
        recs = [d for d in data if d['model_name'] == model]
        if not recs: continue
        
        # Calculate average accuracy
        avg_acc = sum([r['accuracy'] for r in recs]) / len(recs) * 100
        params_million = recs[0]['params'] / 1e6
        flops_g = recs[0]['approx_flops'] / 1e9
        
        ax3.scatter(params_million, avg_acc, 
                    s=flops_g * 100, # Bubble size
                    color=colors[model], alpha=0.7, label=model, edgecolors='w')
        
        ax3.text(params_million, avg_acc + 0.15, model, fontsize=9, ha='center')

    ax3.set_xlabel('Parameters (Millions)')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Accuracy vs Model Size (Bubble Size represents GFLOPs)')
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # Add legend (move to outside to avoid blocking)
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), title="Models")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_accuracy_vs_size.png"), dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    print(f"Summary plots saved to {output_dir}")

def main():
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    ckpt_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pth')]
    ckpt_files.sort()
    
    print(f"Found {len(ckpt_files)} checkpoints. Output dir: {OUTPUT_DIR}")

    # Load DataLoader (only once)
    print("Loading DataLoaders...")
    dataloaders, _ = create_dataloaders(
        splits_dir=SPLITS_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        img_size=224
    )
    
    for ckpt_file in ckpt_files:
        print("="*60)
        print(f"Processing: {ckpt_file}")
        
        model_arch_key = parse_filename(ckpt_file)
        if not model_arch_key:
            print(f"[Skip] Unknown architecture: {ckpt_file}")
            continue
            
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_file)
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        
        # Infer number of classes
        if 'class_names' in checkpoint:
            num_classes = len(checkpoint['class_names'])
        else:
            # Fault tolerance: try to infer from weight shapes
            try:
                sd = checkpoint['model_state_dict']
                # Common FC layer names
                keys = ['fc.weight', 'classifier.1.weight', 'classifier.3.weight', 'head.weight', 'classifier.2.weight']
                found_key = next((k for k in keys if k in sd), None)
                if found_key:
                    num_classes = sd[found_key].shape[0]
                else:
                    raise ValueError("Cannot find FC weights")
            except:
                print("[Error] Cannot determine num_classes. Skipping.")
                continue

        # Build and load model
        try:
            model = load_model_structure(model_arch_key, num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(DEVICE)
        except Exception as e:
            print(f"[Error] Load failed: {e}")
            continue

        display_name = ckpt_file.replace('_best.pth', '')
        
        print(f"Evaluating {display_name}...")
        
        # Call evaluation function
        # Note: passing output_dir parameter here, ensure engine.py supports it
        # If not supported, modify engine.py or accept output to default 'outputs' directory
        try:
            evaluate_on_test(
                model=model,
                dataloaders=dataloaders,
                device=DEVICE,
                model_name=display_name,
                # output_dir=OUTPUT_DIR # If you modified engine.py, uncomment this line
            )
            
            # Temporary workaround: if engine.py doesn't support output_dir
            # Assuming engine.py still hardcodes to ./outputs
            default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
            if default_out != OUTPUT_DIR and os.path.exists(default_out):
                import shutil
                for f in glob.glob(os.path.join(default_out, f"*{display_name}*")):
                    shutil.move(f, OUTPUT_DIR)
                    
        except TypeError:
            # If engine.py parameter definition not updated, catch TypeError and fallback
            evaluate_on_test(
                model=model,
                dataloaders=dataloaders,
                device=DEVICE,
                model_name=display_name
            )
            # Also try moving files
            default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
            if os.path.exists(default_out):
                import shutil
                for f in glob.glob(os.path.join(default_out, f"*{display_name}*")):
                    shutil.move(f, OUTPUT_DIR)

    # Generate final summary plots
    summarize_and_plot(OUTPUT_DIR)

if __name__ == "__main__":
    main()
