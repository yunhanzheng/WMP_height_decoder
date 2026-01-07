"""
Compare recurrent states from domino vs stripes terrain
Runs collection twice and visualizes together
"""

import subprocess
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist


def main():
    num_steps = int(os.environ.get('TSNE_NUM_STEPS', '200'))
    save_path = os.environ.get('TSNE_SAVE_PATH', 'tsne_terrain_comparison.png')
    perplexity = int(os.environ.get('TSNE_PERPLEXITY', '30'))

    print(f"\n{'='*80}")
    print("t-SNE Terrain Comparison: Domino vs Stripes")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  - Steps per terrain: {num_steps}")
    print(f"  - Output: {save_path}")
    print(f"  - Perplexity: {perplexity}")
    print(f"{'='*80}\n")

    # Temporary files for collected states
    domino_file = '/tmp/domino_states.npz'
    stripes_file = '/tmp/stripes_states.npz'

    # Build base command (preserve other args like --task, --headless, etc.)
    base_cmd = [sys.executable, 'collect_terrain_states.py']

    # Pass through command line arguments (except our custom ones)
    extra_args = []
    skip_next = False
    for i, arg in enumerate(sys.argv[1:]):
        if skip_next:
            skip_next = False
            continue
        if arg.startswith('--'):
            extra_args.append(arg)
            # Check if next arg is the value
            if i + 1 < len(sys.argv) - 1 and not sys.argv[i + 2].startswith('--'):
                extra_args.append(sys.argv[i + 2])
                skip_next = True

    # ========================================================================
    # Collect from DOMINO terrain
    # ========================================================================
    print(f"{'='*80}")
    print("STEP 1/3: Collecting from DOMINO terrain")
    print(f"{'='*80}\n")

    domino_env = os.environ.copy()
    domino_env['TERRAIN_TYPE'] = 'domino'
    domino_env['OUTPUT_FILE'] = domino_file
    domino_env['NUM_STEPS'] = str(num_steps)

    domino_cmd = base_cmd + extra_args

    print(f"Running: {' '.join(domino_cmd)}")
    print(f"  TERRAIN_TYPE=domino")
    print(f"  OUTPUT_FILE={domino_file}")
    print(f"  NUM_STEPS={num_steps}\n")

    result = subprocess.run(domino_cmd, env=domino_env)

    if result.returncode != 0:
        print(f"ERROR: Failed to collect domino terrain states")
        sys.exit(1)

    # ========================================================================
    # Collect from STRIPES terrain
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 2/3: Collecting from STRIPES terrain")
    print(f"{'='*80}\n")

    stripes_env = os.environ.copy()
    stripes_env['TERRAIN_TYPE'] = 'stripes'
    stripes_env['OUTPUT_FILE'] = stripes_file
    stripes_env['NUM_STEPS'] = str(num_steps)

    stripes_cmd = base_cmd + extra_args

    print(f"Running: {' '.join(stripes_cmd)}")
    print(f"  TERRAIN_TYPE=stripes")
    print(f"  OUTPUT_FILE={stripes_file}")
    print(f"  NUM_STEPS={num_steps}\n")

    result = subprocess.run(stripes_cmd, env=stripes_env)

    if result.returncode != 0:
        print(f"ERROR: Failed to collect stripes terrain states")
        sys.exit(1)

    # ========================================================================
    # Load and combine
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 3/3: Computing t-SNE and visualizing")
    print(f"{'='*80}\n")

    # Load collected states
    domino_data = np.load(domino_file)
    stripes_data = np.load(stripes_file)

    states_domino = domino_data['states']
    states_stripes = stripes_data['states']

    print(f"Loaded data:")
    print(f"  - Domino: {states_domino.shape[0]} samples")
    print(f"  - Stripes: {states_stripes.shape[0]} samples")

    # Combine
    all_states = np.concatenate([states_domino, states_stripes], axis=0)
    terrain_labels = np.concatenate([
        np.zeros(states_domino.shape[0]),
        np.ones(states_stripes.shape[0])
    ])

    print(f"  - Total: {all_states.shape[0]} samples\n")

    # Compute t-SNE
    n_samples = all_states.shape[0]
    max_perplexity = (n_samples - 1) // 3
    if perplexity >= n_samples:
        perplexity = min(30, max_perplexity)
        print(f"Adjusted perplexity to {perplexity}")

    print(f"Running t-SNE (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000,
                random_state=42, verbose=1)
    tsne_results = tsne.fit_transform(all_states)

    # Visualize
    print(f"\nCreating visualization...")
    fig, ax = plt.subplots(figsize=(12, 8))

    mask_domino = terrain_labels == 0
    mask_stripes = terrain_labels == 1

    ax.scatter(tsne_results[mask_domino, 0], tsne_results[mask_domino, 1],
               c='blue', alpha=0.6, s=20, label='Domino Terrain')
    ax.scatter(tsne_results[mask_stripes, 0], tsne_results[mask_stripes, 1],
               c='red', alpha=0.6, s=20, label='Stripes Terrain')

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('World Model Recurrent States: Domino vs Stripes Terrain',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved to: {save_path}")

    # Statistics
    print(f"\n{'='*80}")
    print("Analysis:")
    print(f"{'='*80}")

    centroid_domino = tsne_results[mask_domino].mean(axis=0)
    centroid_stripes = tsne_results[mask_stripes].mean(axis=0)
    separation = np.linalg.norm(centroid_domino - centroid_stripes)

    print(f"Centroid separation: {separation:.2f}")
    print(f"Avg distance within domino: {np.mean(cdist(tsne_results[mask_domino], tsne_results[mask_domino])):.2f}")
    print(f"Avg distance within stripes: {np.mean(cdist(tsne_results[mask_stripes], tsne_results[mask_stripes])):.2f}")

    if separation > 10:
        print("\n✓ Strong separation: World model clearly distinguishes terrains")
    elif separation > 5:
        print("\n~ Moderate separation: Some terrain-specific patterns")
    else:
        print("\n✗ Weak separation: Similar representations for both terrains")

    # Cleanup
    os.remove(domino_file)
    os.remove(stripes_file)

    print(f"\n{'='*80}")
    print("Done!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
