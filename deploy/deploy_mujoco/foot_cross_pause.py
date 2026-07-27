"""G1 stage-2 style pause: stop when a foot crosses a stripe and lands."""

from typing import List, Sequence, Tuple

import numpy as np

Stripe = Tuple[float, float]  # (center_x, height_m)


class FootCrossPause:
    """Zero velocity command for stop_time after foot-cross rising edge."""

    def __init__(
        self,
        stripes: Sequence[Stripe],
        stop_time_range: Tuple[float, float] = (2.0, 4.0),
        stripe_half_width: float = 0.05,
        rng: np.random.Generator = None,
    ):
        self.stripes = list(stripes)
        self.stop_time_range = stop_time_range
        self.stripe_half_width = stripe_half_width
        self.rng = rng or np.random.default_rng()

        self.cmd_phase_moving = True
        self.pause_timer = 0.0
        self.prev_any_crossed = False
        self.active_stripe_x = 0.0
        # After pause: wait until both feet clear stripe before arming next pause.
        self.post_pause_clear = False

    def _terrain_h(self, x: float) -> float:
        h = 0.0
        for sx, sh in self.stripes:
            if abs(x - sx) <= self.stripe_half_width:
                h = max(h, sh)
        return h

    def _foot_crossed_landed(self, foot_x: float, foot_z: float) -> Tuple[bool, float]:
        """Return (crossed_landed, stripe_center_x behind foot)."""
        contact = foot_z < 0.08
        on_flat = self._terrain_h(foot_x) < 0.03
        max_behind = 0.0
        stripe_x = foot_x
        for d in (0.05, 0.10, 0.15, 0.20, 0.25, 0.35):
            h = self._terrain_h(foot_x - d)
            if h > max_behind:
                max_behind = h
                stripe_x = foot_x - d
        stripe_behind = max_behind > 0.03
        crossed = contact and on_flat and stripe_behind
        return crossed, stripe_x

    def _both_feet_past_stripe(self, foot_xy: List[Tuple[float, float]]) -> bool:
        if self.active_stripe_x <= 0.0:
            return True
        return all(fx > self.active_stripe_x + 0.1 for fx, _ in foot_xy)

    def update(
        self,
        foot_xy: List[Tuple[float, float]],
        dt: float,
        base_cmd: np.ndarray,
    ) -> np.ndarray:
        """Return command fed to policy (zeros while paused)."""
        if not self.stripes:
            return base_cmd

        crossed = []
        stripe_xs = []
        for fx, fz in foot_xy:
            c, sx = self._foot_crossed_landed(fx, fz)
            crossed.append(c)
            stripe_xs.append(sx)
        any_crossed = any(crossed)

        if self.post_pause_clear and self._both_feet_past_stripe(foot_xy):
            self.post_pause_clear = False

        if self.cmd_phase_moving:
            rising = (
                any_crossed
                and not self.prev_any_crossed
                and not self.post_pause_clear
            )
            self.prev_any_crossed = any_crossed

            if rising:
                self.cmd_phase_moving = False
                lo, hi = self.stop_time_range
                self.pause_timer = float(self.rng.uniform(lo, hi))
                # Stripe x from terrain behind the most forward crossed foot.
                scores = [
                    (fx, sx) for (fx, _), c, sx in zip(foot_xy, crossed, stripe_xs) if c
                ]
                if scores:
                    self.active_stripe_x = max(scores, key=lambda t: t[0])[1]
                print(
                    f"[pause] foot crossed stripe @ x≈{self.active_stripe_x:.2f}m, "
                    f"hold {self.pause_timer:.1f}s"
                )
                return np.zeros(3, dtype=np.float32)
            return base_cmd

        # Hold still during pause; keep prev_any_crossed in sync (training does this).
        self.prev_any_crossed = any_crossed
        self.pause_timer -= dt
        if self.pause_timer <= 0.0:
            self.cmd_phase_moving = True
            self.post_pause_clear = True
            print(f"[pause] resume forward cmd={base_cmd.tolist()}")
            return base_cmd
        return np.zeros(3, dtype=np.float32)
