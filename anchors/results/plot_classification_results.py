import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

current_path = os.path.dirname(os.path.abspath(__file__))

# Adjust the file patterns if your CSV files do not have "_base_" in the name.
pattern1 = os.path.join(current_path, 'results_kmeans_mahalanobis_*.csv')
pattern2 = os.path.join(current_path, 'results_random_cosine_*.csv')
files1 = sorted(glob.glob(pattern1))
files2 = sorted(glob.glob(pattern2))

transformer_names = [
    "vit_base_patch16_224",
    "rexnet_100",
    "vit_base_resnet50_384",
    "vit_small_patch16_224"
]

# 3. Load each pair of CSVs into a single DataFrame per base
data = {}
for base in transformer_names:
    f1 = os.path.join(current_path, f'results_kmeans_mahalanobis_base_{base}.csv')
    f2 = os.path.join(current_path, f'results_random_cosine_base_{base}.csv')
    df1 = pd.read_csv(f1); df1['run'] = 'kmeans + mahalanobis'
    df2 = pd.read_csv(f2); df2['run'] = 'random + cosine'
    df = pd.concat([df1, df2], ignore_index=True).sort_values('Target')
    data[base] = df

# Print table in the format shown in the image
print("Table: Performance comparison with different encoding techniques.")
print("The table reports the mean weighted F1 (± std) across different datasets and seeds.")
print()

# Create header
print(f"{'Decoder':<25} {'Encoder':<25} {'rand & cossim':<15} {'kmeans & mah':<15}")
print("-" * 80)

# For each decoder (transformer base)
for decoder in transformer_names:
    df = data[decoder]
    
    # Group by target datasets
    targets = sorted(df['Target'].unique())
    
    # Print the first row with decoder name
    first_row = True
    for target in targets:
        # Get data for this target
        target_data = df[df['Target'] == target]
        
        # Get results for both runs
        abs_data = target_data[target_data['run'] == 'random + cosine']
        rel_data = target_data[target_data['run'] == 'kmeans + mahalanobis']
        
        # Format the results
        if len(abs_data) > 0:
            abs_mean = abs_data['Mean_F1'].iloc[0]
            abs_std = abs_data['Std_F1'].iloc[0]
            abs_str = f"{abs_mean:.2f} ± {abs_std:.2f}"
        else:
            abs_str = "-"
            
        if len(rel_data) > 0:
            rel_mean = rel_data['Mean_F1'].iloc[0]
            rel_std = rel_data['Std_F1'].iloc[0]
            rel_str = f"{rel_mean:.2f} ± {rel_std:.2f}"
        else:
            rel_str = "-"
        
        # Print the row
        if first_row:
            print(f"{decoder:<25} {target:<25} {abs_str:<15} {rel_str:<15}")
            first_row = False
        else:
            print(f"{'':<25} {target:<25} {abs_str:<15} {rel_str:<15}")
    
    print()  # Add blank line between decoder groups

print()  # Add blank line before plot

# 4. Plot setup
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.35
colors = ['tab:blue', 'tab:orange']

# 5. Build the grouped barplot
for i, (base, df) in enumerate(data.items()):
    targets = df['Target'].unique()
    n_targets = len(targets)
    group_start = i * (n_targets + 1)
    x_base = np.arange(n_targets) + group_start
    
    # Updated order and color mapping:
    for j, run in enumerate(['random + cosine', 'kmeans + mahalanobis']):
        means = [df[(df['Target']==t)&(df['run']==run)]['Mean_F1'].item() for t in targets]
        errs  = [df[(df['Target']==t)&(df['run']==run)]['Std_F1'].item()  for t in targets]
        x = x_base + (j - 0.5) * bar_width
        color = 'tab:blue' if run=='random + cosine' else 'tab:orange'
        ax.bar(x, means, yerr=errs, width=bar_width,
               label=run if i == 0 else "", color=color, capsize=4)
    
    # 6. Annotate the base transformer name under its group
    center = group_start + (n_targets - 1) / 2
    center = group_start + (n_targets - 1) / 2
    group_max = (df['Mean_F1'] + df['Std_F1']).max()
    ax.text(center, group_max + 2, base, ha='center', va='bottom', fontweight='bold')

# 7. X‑axis ticks: each pair’s central position, with angled target names
all_positions = []
all_labels = []
for i, (base, df) in enumerate(data.items()):
    n = len(df['Target'].unique())
    start = i * (n + 1)
    for j, tgt in enumerate(df['Target'].unique()):
        all_positions.append(start + j)
        all_labels.append(tgt)

ax.set_xticks(all_positions)
ax.set_xticklabels(all_labels, rotation=45, ha='right')

# Set y-axis lower limit to 40
ymin, ymax = ax.get_ylim()
ax.set_ylim(40, ymax)

# 8. Labels, legend, layout
ax.set_ylabel('Mean F1 Score (%)')
ax.set_title('KMeans, Mahalanobis vs Random, Cosine in Zero-Shot Classification')
ax.legend(loc='upper right', bbox_to_anchor=(1, 1.11), ncol=1)

# Adjust subplots to add more top space for the legend and bottom space for tick labels
plt.subplots_adjust(top=0.85, bottom=0.25)
plt.show()