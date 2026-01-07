# t-SNE Visualization of World Model Recurrent States

This directory contains tools for visualizing the recurrent states of your world model using t-SNE dimensionality reduction.

## Overview

The world model in WMP uses an RSSM (Recurrent State Space Model) that maintains:
- **Deterministic state** (`deter`): Output of a GRU cell, shape `[batch, deter_dim]`
- **Stochastic state** (`stoch`): Sampled from a learned distribution, shape `[batch, stoch_dim]` or `[batch, stoch, discrete]`

t-SNE visualization helps you understand:
- How the world model represents different states
- Whether similar states cluster together
- Temporal patterns in the latent space
- Correlation between recurrent states and rewards/behaviors

## Files

1. **visualize_recurrent_state.py** - Core visualization library with `RecurrentStateVisualizer` class
2. **run_tsne_visualization.py** - Standalone script to visualize from a checkpoint
3. **tsne_integration_example.py** - Example of integrating into training loop

## Quick Start

### Option 1: Visualize from a saved checkpoint

```bash
python run_tsne_visualization.py \
    --task=a1 \
    --checkpoint=logs/model_1000.pt \
    --num_steps=1000 \
    --color_by=reward \
    --save_path=tsne_visualization.png
```

Arguments:
- `--task`: Your task name (e.g., `a1`, `go1`)
- `--checkpoint`: Path to your model checkpoint (.pt file)
- `--num_steps`: Number of environment steps to collect (default: 1000)
- `--num_envs`: Number of parallel environments (default: 100)
- `--use_full_state`: Include both deterministic and stochastic states (default: deter only)
- `--color_by`: Color points by `timestep`, `reward`, `episode`, or `none`
- `--save_path`: Where to save the output image
- `--perplexity`: t-SNE perplexity parameter (default: 30)

### Option 2: Integrate into training loop

In your training script:

```python
from tsne_integration_example import add_tsne_visualization_to_runner

# After creating your runner
runner = WMPRunner(env, train_cfg, log_dir, device)
add_tsne_visualization_to_runner(runner)

# During training (in the learn loop)
if iteration % 100 == 0:
    runner.visualize_recurrent_states(
        num_steps=500,
        color_by='reward',
        save_path=f'tsne_iter_{iteration}.png'
    )
```

### Option 3: Use the API directly

```python
from visualize_recurrent_state import RecurrentStateVisualizer

# Create visualizer
visualizer = RecurrentStateVisualizer(use_deter_only=True)

# During your rollout/training loop
for step in range(num_steps):
    # ... your world model forward pass ...
    wm_embed = world_model.encoder(wm_obs)
    wm_latent, _ = world_model.dynamics.obs_step(wm_latent, wm_action, wm_embed, is_first)

    # Collect the state
    visualizer.collect_state(
        wm_latent,
        reward=rewards,
        timestep=step,
        episode=episode_ids
    )

# Compute t-SNE
tsne_results = visualizer.compute_tsne(perplexity=30, n_iter=1000)

# Plot
visualizer.plot_tsne(tsne_results, color_by='reward', save_path='output.png')
```

## Understanding the Results

### Coloring Options

- **timestep**: Shows temporal progression through the episode
  - Useful for understanding how states evolve over time
  - Look for smooth transitions vs. jumps

- **reward**: Shows correlation between latent states and rewards
  - Green regions = high reward states
  - Red regions = low reward states
  - Useful for understanding if the model captures reward-relevant features

- **episode**: Different colors for different parallel environments
  - Useful for understanding if different environments produce similar state representations
  - Can reveal if certain environments are outliers

### What to Look For

1. **Clustering**: Do similar behaviors/states cluster together?
2. **Continuity**: Are temporally adjacent states close in t-SNE space?
3. **Reward correlation**: Do high-reward states cluster separately from low-reward ones?
4. **Coverage**: Does the model explore a diverse range of states?

## Advanced Usage

### Custom Analysis

You can access the collected data for custom analysis:

```python
visualizer = RecurrentStateVisualizer()
# ... collect states ...

# Access raw data
all_states = np.concatenate(visualizer.states, axis=0)
all_rewards = np.concatenate(visualizer.metadata['rewards'])
all_timesteps = np.concatenate(visualizer.metadata['timesteps'])

# Your custom analysis here
# e.g., PCA, UMAP, clustering, etc.
```

### 3D Visualization

```python
# Compute 3D t-SNE
tsne_3d = visualizer.compute_tsne(n_components=3)

# Plot in 3D
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tsne_3d[:, 0], tsne_3d[:, 1], tsne_3d[:, 2], c=rewards, cmap='RdYlGn')
plt.show()
```

### Comparing Different Checkpoints

```python
# Visualize states from multiple checkpoints
checkpoints = ['model_100.pt', 'model_500.pt', 'model_1000.pt']
for i, ckpt in enumerate(checkpoints):
    visualizer, tsne = visualize_from_checkpoint(
        checkpoint_path=ckpt,
        runner=runner,
        save_path=f'tsne_checkpoint_{i}.png'
    )
```

## Tips

1. **Perplexity**: Start with 30, increase for larger datasets (>5000 samples), decrease for smaller
2. **Number of steps**: 500-2000 steps usually provides good coverage
3. **Use deterministic state first**: Easier to interpret than full state
4. **Multiple runs**: t-SNE has randomness, run multiple times to verify patterns
5. **Color by reward**: Most informative for understanding if model learns meaningful representations

## Troubleshooting

**Memory issues**: Reduce `num_steps` or `num_envs`

**t-SNE takes too long**: Reduce `n_iter` (try 500) or number of samples

**Plot looks like noise**: Try adjusting perplexity, or collect more diverse states

**States don't separate**: Model may need more training, or states may genuinely be similar

## Example Output

The visualization will show a 2D scatter plot where:
- Each point represents the recurrent state at one timestep in one environment
- Colors indicate the value of the selected metadata (timestep/reward/episode)
- Nearby points represent similar internal representations
- Clusters suggest the model groups similar situations together

This helps you understand what your world model has learned and how it organizes information internally.
