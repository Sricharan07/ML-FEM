#!/usr/bin/env python3
"""Run the logic.py notch demo and export its plots to PNG files."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from logic import run_demo_simulation


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    counter = {"idx": 0}

    def save_and_close(*args, **kwargs):
        counter["idx"] += 1
        filename = output_dir / f"logic_plot_{counter['idx']}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close("all")

    plt.show = save_and_close  # type: ignore[assignment]

    run_demo_simulation()
    print(f"[export] Saved {counter['idx']} figure(s) to {output_dir}")


if __name__ == "__main__":
    main()
