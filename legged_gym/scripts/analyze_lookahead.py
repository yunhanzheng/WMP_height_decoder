"""
Cross-correlation look-ahead analysis for World Model Predictive control.

Usage
-----
  # play.py --visualize_sensitivity calls plot_lookahead() directly at the end
  # of the run — no file is written.

  # To use standalone, supply a pre-saved .npz (fl_contact_z, rl_foot_z, dt,
  # first_event_step arrays):
  python analyze_lookahead.py --npz /path/to/signals.npz

  # Or import and call directly:
  #   from legged_gym.scripts.analyze_lookahead import plot_lookahead
  #   plot_lookahead(fl_contact_z, rl_foot_z, dt=0.02, first_event_step=120)

Interpretation
--------------
  The robot hits an obstacle with its FL foot at t = first_event_step.
  Because the World Model encodes terrain context, the robot begins lifting
  its RL foot *before* the RL foot physically reaches the obstacle.
  The CCF peak lag  tau* < 0  (or equivalently  tau* > 0  depending on
  convention — see the plot title) quantifies this predictive look-ahead.
"""

import argparse
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sp_signal


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def _standardize(x: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalisation."""
    std = x.std()
    return (x - x.mean()) / (std if std > 1e-9 else 1.0)


def compute_ccf(front: np.ndarray, rear: np.ndarray, max_lag: Optional[int] = None):
    """
    Compute the normalised cross-correlation function (CCF).

    CCF[k]  =  correlation of rear[t]  with  front[t - k]
             =  how much rear at time t looks like front k steps earlier.

    A peak at k > 0 means rear lags front by k steps
    (rear reacts k steps after front — pure reactive).
    A peak at k < 0 means rear leads front by |k| steps
    (rear moves before front reaches the landmark — predictive / look-ahead).

    Parameters
    ----------
    front, rear : 1-D arrays of equal length, already standardised.
    max_lag     : only return lags in [-max_lag, +max_lag].

    Returns
    -------
    lags : int array  shape (2*T-1,) or (2*max_lag+1,)
    ccf  : float array, same shape, values in [-1, 1]
    """
    T = len(front)
    if max_lag is None:
        max_lag = T - 1

    full = sp_signal.correlate(rear, front, mode='full')   # length 2T-1
    norm = np.sqrt(
        sp_signal.correlate(front, front, mode='valid')[0] *
        sp_signal.correlate(rear,  rear,  mode='valid')[0]
    )
    full = full / (norm if norm > 1e-12 else 1.0)

    lag_full = sp_signal.correlation_lags(T, T, mode='full')

    mask = (lag_full >= -max_lag) & (lag_full <= max_lag)
    return lag_full[mask], full[mask]


# ---------------------------------------------------------------------------
# Main plotting function (importable)
# ---------------------------------------------------------------------------

def plot_lookahead(
    fl_foot_z: np.ndarray,
    rl_foot_z: np.ndarray,
    dt: float = 0.02,
    first_event_step: Optional[int] = None,
    max_lag: int = 60,
    save_path: Optional[str] = None,
):
    """
    Two-panel figure:
      Panel 1 — time-series overlay (normalised FL foot height vs RL foot height)
      Panel 2 — Cross-Correlation Function with peak annotation

    Parameters
    ----------
    fl_foot_z       : FL foot world-frame z position, one value per sim step.
    rl_foot_z       : RL foot world-frame z position, one value per sim step.
    dt              : simulation time-step in seconds (for axis labelling).
    first_event_step: step index when FL hits the obstacle (red vertical line).
    max_lag         : maximum lag shown in CCF panel (frames).
    """
    T = min(len(fl_foot_z), len(rl_foot_z))
    fl = fl_foot_z[:T].astype(float)
    rl = rl_foot_z[:T].astype(float)

    fl_norm = _standardize(fl)
    rl_norm = _standardize(rl)

    lags, ccf = compute_ccf(fl_norm, rl_norm, max_lag=max_lag)

    peak_idx  = int(np.argmax(ccf))
    peak_lag  = int(lags[peak_idx])
    peak_corr = float(ccf[peak_idx])

    steps = np.arange(T)

    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(11, 8),
                             gridspec_kw={'hspace': 0.45})

    # --- Panel 1: time series -----------------------------------------------
    ax1 = axes[0]
    ax1.plot(steps, fl_norm, color='steelblue',  lw=1.2, label='FL foot height z (norm.)')
    ax1.plot(steps, rl_norm, color='darkorange', lw=1.2, label='RL foot height z (norm.)')

    if first_event_step is not None and 0 <= first_event_step < T:
        ax1.axvline(x=first_event_step, color='red', lw=1.5, ls='--',
                    label=f'FL hits obstacle  (t={first_event_step})')

    # Annotate where the RL lift starts (peak lag relative to event)
    if first_event_step is not None and peak_lag != 0:
        rl_lift_step = first_event_step + peak_lag
        if 0 <= rl_lift_step < T:
            ax1.axvline(x=rl_lift_step, color='green', lw=1.5, ls='--',
                        label=f'RL lift predicted  (t={rl_lift_step},  lag={peak_lag:+d})')

    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Normalised value')
    ax1.set_title('Time Series: FL foot height vs RL foot height')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Secondary x-axis in seconds
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    tick_steps = ax1.get_xticks()
    ax1_top.set_xticks(tick_steps)
    ax1_top.set_xticklabels([f'{s * dt:.2f}' for s in tick_steps], fontsize=7)
    ax1_top.set_xlabel('Time (s)', fontsize=8)

    # --- Panel 2: CCF -------------------------------------------------------
    ax2 = axes[1]
    ax2.bar(lags, ccf, width=1.0, color='slategray', alpha=0.6, label='CCF')
    ax2.axvline(x=0,        color='black',  lw=0.8, ls=':')
    ax2.axhline(y=0,        color='black',  lw=0.8, ls=':')
    ax2.axvline(x=peak_lag, color='crimson', lw=2.0,
                label=f'Peak  τ = {peak_lag:+d} frames  ({peak_lag * dt:+.3f} s),  r = {peak_corr:.3f}')

    # Shade the look-ahead region
    if peak_lag < 0:
        ax2.axvspan(peak_lag, 0, alpha=0.12, color='green',
                    label='Look-ahead region  (RL lifts before FL landmark)')
        direction_note = (
            f"Peak lag τ* = {peak_lag:+d} frames  →  RL foot leads FL by |τ*| = {abs(peak_lag)} steps "
            f"({abs(peak_lag) * dt:.3f} s).\n"
            "Negative lag: the World Model enables proactive RL lift BEFORE the foot reaches the obstacle."
        )
    elif peak_lag > 0:
        ax2.axvspan(0, peak_lag, alpha=0.12, color='orange',
                    label='Reactive region  (RL lifts after FL landmark)')
        direction_note = (
            f"Peak lag τ* = {peak_lag:+d} frames  →  RL foot lags FL by τ* = {peak_lag} steps "
            f"({peak_lag * dt:.3f} s).\n"
            "Positive lag: RL foot reacts AFTER the front foot hits the obstacle (purely reactive)."
        )
    else:
        direction_note = "Peak lag τ* = 0: RL and FL signals are synchronous."

    ax2.set_xlabel('Lag τ (frames)  [positive = rear lags front, negative = rear leads front]')
    ax2.set_ylabel('Correlation coefficient')
    ax2.set_title(
        'Cross-Correlation Function (CCF):  CCF[τ] = corr( RL_height[t],  FL_height[t − τ] )\n'
        'Peak lag τ* quantifies the World Model look-ahead',
        fontsize=9
    )
    ax2.legend(fontsize=8, loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Annotation box
    fig.text(0.5, 0.01, direction_note, ha='center', va='bottom', fontsize=8,
             style='italic',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('World Model Look-Ahead Analysis: FL→RL Cross-Correlation', fontsize=12, y=1.01)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved look-ahead CCF figure to: {save_path}")
    plt.show()

    # Summary
    print("\n=== Look-Ahead Cross-Correlation Summary ===")
    print(f"  Signal length      : {T} steps  ({T * dt:.2f} s)")
    print(f"  Peak CCF lag  τ*   : {peak_lag:+d} frames  ({peak_lag * dt:+.4f} s)")
    print(f"  Peak correlation   : {peak_corr:.4f}")
    if first_event_step is not None:
        rl_pred = first_event_step + peak_lag
        print(f"  FL hits obstacle @ : step {first_event_step}  ({first_event_step * dt:.3f} s)")
        print(f"  RL predicted lift  : step {rl_pred}  ({rl_pred * dt:.3f} s)")
    print(direction_note)


# ---------------------------------------------------------------------------
# Comparison plot: WM vs baseline overlaid
# ---------------------------------------------------------------------------

def plot_comparison(
    wm_npz: str,
    baseline_npz: str,
    max_lag: int = 60,
    save_path: Optional[str] = None,
):
    """
    Overlay CCFs from two policies (WM and baseline) on one figure.

    Top panel    — time series of both policies (FL and RL foot heights)
    Bottom panel — CCF of WM (blue) vs CCF of baseline (orange) overlaid,
                   with vertical lines at each peak lag.
    """
    def _load(path):
        d = np.load(path)
        fl = d['fl_foot_z'].astype(float)
        rl = d['rl_foot_z'].astype(float)
        dt = float(d['dt'])
        fes = int(d['first_event_step'])
        return fl, rl, dt, (fes if fes >= 0 else None)

    fl_wm,   rl_wm,   dt, fes_wm   = _load(wm_npz)
    fl_base, rl_base, _,  fes_base = _load(baseline_npz)

    def _ccf_for(fl, rl):
        T = min(len(fl), len(rl))
        fl_n = _standardize(fl[:T])
        rl_n = _standardize(rl[:T])
        lags, ccf = compute_ccf(fl_n, rl_n, max_lag=max_lag)
        peak_idx = int(np.argmax(ccf))
        return fl_n, rl_n, lags, ccf, int(lags[peak_idx]), float(ccf[peak_idx]), T

    fl_wm_n,   rl_wm_n,   lags, ccf_wm,   peak_wm,   r_wm,   T_wm   = _ccf_for(fl_wm,   rl_wm)
    fl_base_n, rl_base_n, _,    ccf_base,  peak_base, r_base, T_base = _ccf_for(fl_base, rl_base)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={'hspace': 0.45})

    # --- Panel 1: time series (WM only, for reference) ----------------------
    ax1 = axes[0]
    steps_wm   = np.arange(T_wm)
    steps_base = np.arange(T_base)
    ax1.plot(steps_wm,   fl_wm_n,   color='steelblue',   lw=1.0, label='WM — FL foot height')
    ax1.plot(steps_wm,   rl_wm_n,   color='steelblue',   lw=1.0, ls='--', alpha=0.7, label='WM — RL foot height')
    ax1.plot(steps_base, fl_base_n, color='darkorange',  lw=1.0, label='Baseline — FL foot height')
    ax1.plot(steps_base, rl_base_n, color='darkorange',  lw=1.0, ls='--', alpha=0.7, label='Baseline — RL foot height')
    if fes_wm is not None:
        ax1.axvline(x=fes_wm,   color='steelblue',  lw=1.5, ls=':', label=f'WM FL hits obstacle (t={fes_wm})')
    if fes_base is not None:
        ax1.axvline(x=fes_base, color='darkorange', lw=1.5, ls=':', label=f'Baseline FL hits obstacle (t={fes_base})')
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Normalised value')
    ax1.set_title('Time Series: FL & RL foot heights — WM vs Baseline')
    ax1.legend(fontsize=7, loc='upper right', ncol=2)
    ax1.grid(True, alpha=0.3)

    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    tick_steps = ax1.get_xticks()
    ax1_top.set_xticks(tick_steps)
    ax1_top.set_xticklabels([f'{s * dt:.2f}' for s in tick_steps], fontsize=7)
    ax1_top.set_xlabel('Time (s)', fontsize=8)

    # --- Panel 2: CCF overlay -----------------------------------------------
    ax2 = axes[1]
    ax2.bar(lags, ccf_wm,   width=1.0, color='steelblue',  alpha=0.5, label='CCF — WM')
    ax2.bar(lags, ccf_base, width=1.0, color='darkorange',  alpha=0.5, label='CCF — Baseline')
    ax2.axvline(x=0, color='black', lw=0.8, ls=':')
    ax2.axhline(y=0, color='black', lw=0.8, ls=':')
    ax2.axvline(x=peak_wm,   color='steelblue',  lw=2.0,
                label=f'WM peak      τ* = {peak_wm:+d} frames ({peak_wm * dt:+.3f} s),  r = {r_wm:.3f}')
    ax2.axvline(x=peak_base, color='darkorange', lw=2.0,
                label=f'Baseline peak τ* = {peak_base:+d} frames ({peak_base * dt:+.3f} s),  r = {r_base:.3f}')

    shift = peak_base - peak_wm
    note = (
        f"WM τ* = {peak_wm:+d} frames   |   Baseline τ* = {peak_base:+d} frames   |   "
        f"Δτ = {shift:+d} frames ({shift * dt:+.3f} s)\n"
        + ("WM RL foot leads baseline by Δτ — World Model look-ahead advantage."
           if shift > 0 else
           "Baseline RL foot leads WM by |Δτ| — no look-ahead advantage detected.")
    )

    ax2.set_xlabel('Lag τ (frames)  [positive = RL lags FL, negative = RL leads FL]')
    ax2.set_ylabel('Correlation coefficient')
    ax2.set_title(
        'CCF Comparison: WM vs Baseline\n'
        'CCF[τ] = corr( RL_height[t],  FL_height[t − τ] )',
        fontsize=9
    )
    ax2.legend(fontsize=8, loc='upper left')
    ax2.grid(True, alpha=0.3)

    fig.text(0.5, 0.01, note, ha='center', va='bottom', fontsize=8, style='italic',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))
    fig.suptitle('World Model vs Baseline: FL→RL Look-Ahead Comparison', fontsize=12, y=1.01)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison figure to: {save_path}")
    plt.show()

    print("\n=== CCF Comparison Summary ===")
    print(f"  WM       peak τ* : {peak_wm:+d} frames  ({peak_wm * dt:+.4f} s),  r = {r_wm:.4f}")
    print(f"  Baseline peak τ* : {peak_base:+d} frames  ({peak_base * dt:+.4f} s),  r = {r_base:.4f}")
    print(f"  Δτ (WM − Baseline): {peak_wm - peak_base:+d} frames  ({(peak_wm - peak_base) * dt:+.4f} s)")
    print(note)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Cross-correlation look-ahead analysis for WMP robot.')
    parser.add_argument('--npz', required=True,
                        help='Path to lookahead_signals.npz saved by play.py')
    parser.add_argument('--max_lag', type=int, default=60,
                        help='Maximum lag (frames) shown in CCF (default: 60)')
    args = parser.parse_args()

    data = np.load(args.npz)
    fl_contact_z     = data['fl_contact_z']
    rl_foot_z        = data['rl_foot_z']
    dt               = float(data['dt'])
    first_event_step_val = int(data['first_event_step'])
    first_event_step = first_event_step_val if first_event_step_val >= 0 else None

    plot_lookahead(
        fl_contact_z=fl_contact_z,
        rl_foot_z=rl_foot_z,
        dt=dt,
        first_event_step=first_event_step,
        max_lag=args.max_lag,
    )


if __name__ == '__main__':
    main()
