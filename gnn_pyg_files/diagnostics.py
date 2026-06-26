from __future__ import annotations
import csv
import os
from typing import Optional, Sequence

import torch


def extract_value(out) -> torch.Tensor:
    """Pull the scalar value/cost prediction out of a network forward pass.

    The network's output interface isn't fixed here, so this tries the common
    shapes. If your value head uses a different key/position, edit THIS function
    only -- everything downstream consumes a flat (num_graphs,) tensor.
    """
    if isinstance(out, dict):
        for k in ("value", "values", "cost", "value_pred", "y_value"):
            if k in out:
                v = out[k]
                break
        else:
            raise KeyError(f"no value-like key in model output; keys={list(out)}")
    elif isinstance(out, (tuple, list)):
        v = out[-1]                       # assume (policy, value)
    else:
        v = getattr(out, "value", out)
    return v.detach().reshape(-1).float().cpu()


class Diagnostics:
    """Minimal CSV + plot logger for self-play diagnostics.

    Writes two CSVs (episode_quality.csv, value_calibration.csv) incrementally,
    and on finish() emits optimality_gap.png and value_calibration.png.
    All counts are in inserted rays (= stellar subdivisions applied).

    NOTE on censoring: a timed-out episode reports agent_rays = max_steps, so its
    gap is a lower bound, not a true optimality gap. The optimality-gap plot here
    uses resolved episodes only and marks timeouts separately. For the full
    multi-panel report use read_diagnostics.py on the CSVs.
    """

    def __init__(self, outdir: str = "diagnostics", rolling: int = 20):
        os.makedirs(outdir, exist_ok=True)
        self.outdir = outdir
        self.rolling = rolling
        self._calib_warned = False

        self.episode_path = os.path.join(outdir, "episode_quality.csv")
        self.calib_path   = os.path.join(outdir, "value_calibration.csv")

        self._ep_f = open(self.episode_path, "w", newline="")
        self._ep_w = csv.writer(self._ep_f)
        self._ep_w.writerow(
            ["episode", "n", "mult", "agent_rays", "baseline_rays", "gap", "resolved"]
        )

        self._cal_f = open(self.calib_path, "w", newline="")
        self._cal_w = csv.writer(self._cal_f)
        self._cal_w.writerow(["episode", "state_idx", "predicted_value", "target_value"])

        # in-memory buffers for plotting
        self._episodes: list[int]   = []
        self._gaps: list[float]     = []
        self._resolved: list[bool]  = []          # NEW: lets plots drop censored timeouts
        self._cal_pred: list[float] = []
        self._cal_tgt: list[float]  = []
        self._cal_ep: list[int]     = []          # NEW: episode per calibration point

    # ----- per-episode solution quality -----
    def log_episode(self, *, episode: int, n: int, mult: int,
                    agent_rays: int, baseline_rays: int, resolved: bool) -> None:
        gap = agent_rays - baseline_rays          # < 0 means agent beat the min_sum baseline
        self._ep_w.writerow([episode, n, mult, agent_rays, baseline_rays, gap, int(resolved)])
        self._ep_f.flush()
        self._episodes.append(episode)
        self._gaps.append(float(gap))
        self._resolved.append(bool(resolved))

    # ----- value-head calibration -----
    def log_calibration(self, *, episode: int,
                        predicted: Sequence[float], target: Sequence[float]) -> None:
        for i, (p, t) in enumerate(zip(predicted, target)):
            self._cal_w.writerow([episode, i, float(p), float(t)])
            self._cal_pred.append(float(p))
            self._cal_tgt.append(float(t))
            self._cal_ep.append(int(episode))
        self._cal_f.flush()

    def warn_calibration_once(self, err: Exception) -> None:
        if not self._calib_warned:
            print(f"[diagnostics] value-head calibration disabled: {err!r}\n"
                  f"              -> fix extract_value() in diagnostics.py to match your value head.")
            self._calib_warned = True

    # ----- plots -----
    def _rolling_mean(self, xs: list[float]) -> list[float]:
        w, out = self.rolling, []
        for i in range(len(xs)):
            lo = max(0, i - w + 1)
            out.append(sum(xs[lo:i + 1]) / (i - lo + 1))
        return out

    def plot_optimality_gap(self, out_png: Optional[str] = None) -> Optional[str]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[diagnostics] matplotlib not installed; "
                  f"curve skipped. Data is in {self.episode_path}")
            return None
        if not self._episodes:
            return None

        out_png = out_png or os.path.join(self.outdir, "optimality_gap.png")

        # resolved-only: timed-out gaps are censored lower bounds, not real gaps
        xr = [e for e, r in zip(self._episodes, self._resolved) if r]
        gr = [g for g, r in zip(self._gaps, self._resolved) if r]
        xt = [e for e, r in zip(self._episodes, self._resolved) if not r]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axhline(0.0, color="gray", lw=1, ls="--", label="min_sum baseline (gap = 0)")
        if xr:
            ax.scatter(xr, gr, s=12, alpha=0.3, color="C0", label="resolved")
            roll = self._rolling_mean([float(g) for g in gr])
            ax.plot(xr, roll, color="C1", lw=2, label=f"rolling mean (w={self.rolling})")
        if xt:
            ymin = min(gr) if gr else 0.0
            ax.scatter(xt, [ymin - 0.5] * len(xt), s=10, marker="x", color="C3",
                       alpha=0.5, label="timed out")
        ax.set_xlabel("episode")
        ax.set_ylabel("inserted-ray gap  (agent \u2212 min_sum)")
        ax.set_title("Optimality-gap learning curve\n(resolved only; lower is better; < 0 beats baseline)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        return out_png

    def plot_calibration(self, out_png: Optional[str] = None) -> Optional[str]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None
        if not self._cal_pred:
            return None

        out_png = out_png or os.path.join(self.outdir, "value_calibration.png")
        lo = min(min(self._cal_pred), min(self._cal_tgt))
        hi = max(max(self._cal_pred), max(self._cal_tgt))
        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.plot([lo, hi], [lo, hi], color="gray", ls="--", lw=1, label="perfect (y = x)")
        sc = ax.scatter(self._cal_tgt, self._cal_pred, s=10, alpha=0.4,
                        c=self._cal_ep, cmap="viridis")
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("episode")
        ax.set_xlabel("target value")
        ax.set_ylabel("predicted value")
        ax.set_title("Value-head calibration\n(below line = optimistic; color = training time)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        return out_png

    def close(self) -> None:
        self._ep_f.close()
        self._cal_f.close()

    def finish(self) -> dict:
        paths = {
            "optimality_gap": self.plot_optimality_gap(),
            "value_calibration": self.plot_calibration(),
            "episode_csv": self.episode_path,
            "calibration_csv": self.calib_path,
        }
        self.close()
        return paths