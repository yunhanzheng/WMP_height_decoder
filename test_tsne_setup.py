"""
Quick test script to verify t-SNE visualization setup

This creates a simple synthetic example to verify that the visualization code works.
"""

import numpy as np
import matplotlib.pyplot as plt
from visualize_recurrent_state import RecurrentStateVisualizer


def test_basic_functionality():
    """Test basic functionality with synthetic data"""
    print("Testing basic t-SNE visualization functionality...")

    # Create visualizer
    visualizer = RecurrentStateVisualizer(use_deter_only=True)

    # Generate synthetic recurrent states
    # Simulate 3 different "behaviors" with distinct state patterns
    n_steps = 50
    n_envs = 20
    state_dim = 200  # typical deter dimension

    print(f"Generating synthetic data: {n_steps} steps, {n_envs} envs, {state_dim} dims")

    for step in range(n_steps):
        # Create synthetic states with 3 distinct clusters
        states = []
        rewards = []
        episodes = []

        for env_id in range(n_envs):
            # Assign environment to one of 3 behaviors
            behavior = env_id % 3

            # Generate state based on behavior
            if behavior == 0:
                # Behavior 1: Low reward, state centered at [1, 0, ...]
                state = np.random.randn(state_dim) * 0.5
                state[0] += 2.0
                reward = -0.5 + np.random.rand() * 0.3
            elif behavior == 1:
                # Behavior 2: Medium reward, state centered at [0, 1, ...]
                state = np.random.randn(state_dim) * 0.5
                state[1] += 2.0
                reward = 0.0 + np.random.rand() * 0.5
            else:
                # Behavior 3: High reward, state centered at [-1, -1, ...]
                state = np.random.randn(state_dim) * 0.5
                state[0] -= 2.0
                state[1] -= 2.0
                reward = 0.5 + np.random.rand() * 0.5

            states.append(state)
            rewards.append(reward)
            episodes.append(env_id)

        states = np.stack(states)
        rewards = np.array(rewards)
        episodes = np.array(episodes)

        # Create mock wm_latent dict
        wm_latent = {'deter': states}

        # Collect state
        visualizer.collect_state(
            wm_latent,
            reward=rewards,
            timestep=step * np.ones(n_envs),
            episode=episodes
        )

    print(f"Collected {len(visualizer.states)} batches of states")
    total_samples = sum(s.shape[0] for s in visualizer.states)
    print(f"Total samples: {total_samples}")

    # Compute t-SNE
    print("\nComputing t-SNE...")
    tsne_results = visualizer.compute_tsne(perplexity=min(30, total_samples // 4))

    print(f"t-SNE results shape: {tsne_results.shape}")

    # Create visualizations
    print("\nCreating visualizations...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Color by timestep
    timesteps = np.concatenate(visualizer.metadata['timesteps'])
    sc1 = axes[0].scatter(tsne_results[:, 0], tsne_results[:, 1],
                         c=timesteps, cmap='viridis', alpha=0.6, s=20)
    axes[0].set_title('Colored by Timestep')
    axes[0].set_xlabel('t-SNE Dimension 1')
    axes[0].set_ylabel('t-SNE Dimension 2')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(sc1, ax=axes[0], label='Timestep')

    # Plot 2: Color by reward
    rewards = np.concatenate(visualizer.metadata['rewards'])
    sc2 = axes[1].scatter(tsne_results[:, 0], tsne_results[:, 1],
                         c=rewards, cmap='RdYlGn', alpha=0.6, s=20)
    axes[1].set_title('Colored by Reward')
    axes[1].set_xlabel('t-SNE Dimension 1')
    axes[1].set_ylabel('t-SNE Dimension 2')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(sc2, ax=axes[1], label='Reward')

    # Plot 3: Color by episode (behavior)
    episodes = np.concatenate(visualizer.metadata['episodes'])
    sc3 = axes[2].scatter(tsne_results[:, 0], tsne_results[:, 1],
                         c=episodes % 3, cmap='tab10', alpha=0.6, s=20)
    axes[2].set_title('Colored by Behavior Type')
    axes[2].set_xlabel('t-SNE Dimension 1')
    axes[2].set_ylabel('t-SNE Dimension 2')
    axes[2].grid(True, alpha=0.3)
    plt.colorbar(sc3, ax=axes[2], label='Behavior ID')

    plt.tight_layout()
    plt.savefig('test_tsne_synthetic.png', dpi=200, bbox_inches='tight')
    print("Saved test visualization to: test_tsne_synthetic.png")
    plt.show()

    print("\n" + "="*80)
    print("SUCCESS! The t-SNE visualization is working correctly.")
    print("="*80)
    print("\nWhat you should see:")
    print("1. Left plot: Gradual color change from dark to light (timestep progression)")
    print("2. Middle plot: Three distinct clusters (green=high reward, red=low reward)")
    print("3. Right plot: Three distinct clusters colored by behavior type")
    print("\nIf you see these patterns, the visualization is working correctly!")

    return visualizer, tsne_results


def test_with_stochastic_state():
    """Test with both deterministic and stochastic states"""
    print("\n" + "="*80)
    print("Testing with full state (deter + stoch)...")
    print("="*80)

    visualizer = RecurrentStateVisualizer(use_deter_only=False)

    n_steps = 30
    n_envs = 10
    deter_dim = 200
    stoch_dim = 30

    for step in range(n_steps):
        deter_states = np.random.randn(n_envs, deter_dim)
        stoch_states = np.random.randn(n_envs, stoch_dim)

        wm_latent = {
            'deter': deter_states,
            'stoch': stoch_states
        }

        visualizer.collect_state(
            wm_latent,
            timestep=step * np.ones(n_envs)
        )

    print(f"Collected {len(visualizer.states)} batches")
    all_states = np.concatenate(visualizer.states, axis=0)
    print(f"Combined state dimension: {all_states.shape[1]} (deter={deter_dim} + stoch={stoch_dim})")

    tsne_results = visualizer.compute_tsne()
    print(f"t-SNE results shape: {tsne_results.shape}")

    visualizer.plot_tsne(tsne_results, color_by=None,
                        save_path='test_tsne_full_state.png')

    print("Saved full state test to: test_tsne_full_state.png")
    print("SUCCESS! Full state (deter + stoch) visualization works.")

    return visualizer, tsne_results


def check_dependencies():
    """Check if all required packages are installed"""
    print("Checking dependencies...")

    dependencies = {
        'numpy': True,
        'matplotlib': True,
        'sklearn': True,
        'torch': True
    }

    for package in dependencies:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is NOT installed")
            dependencies[package] = False

    if all(dependencies.values()):
        print("\n✓ All dependencies are installed!")
        return True
    else:
        print("\n✗ Some dependencies are missing. Please install them:")
        missing = [pkg for pkg, installed in dependencies.items() if not installed]
        print(f"  pip install {' '.join(missing)}")
        return False


if __name__ == "__main__":
    print("="*80)
    print("t-SNE Visualization Test Suite")
    print("="*80)

    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        exit(1)

    print("\n")

    # Test 1: Basic functionality
    try:
        visualizer1, tsne1 = test_basic_functionality()
        print("\n✓ Test 1 passed: Basic functionality")
    except Exception as e:
        print(f"\n✗ Test 1 failed: {str(e)}")
        import traceback
        traceback.print_exc()

    # Test 2: Full state (deter + stoch)
    try:
        visualizer2, tsne2 = test_with_stochastic_state()
        print("\n✓ Test 2 passed: Full state visualization")
    except Exception as e:
        print(f"\n✗ Test 2 failed: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)
    print("\nNext steps:")
    print("1. Check the generated images: test_tsne_synthetic.png and test_tsne_full_state.png")
    print("2. Run on real data using: python run_tsne_visualization.py --checkpoint <path>")
    print("3. Or integrate into your training loop using tsne_integration_example.py")
