"""
app.py – KNN + DCT Learning GUI
================================
Run with:
    python app.py

Requirements: numpy, matplotlib, tkinter (bundled with most Python installs).

Two tabs are provided:
  • KNN Tab  – visualise the K-Nearest Neighbours decision boundary on a 2-D
               scatter plot.  Use the slider to change k and observe how the
               boundary changes.
  • DCT Tab  – demonstrate 1-D signal compression via the Discrete Cosine
               Transform.  Use the slider to choose how many coefficients to
               keep and see the reconstructed signal alongside the original.
"""

import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # must come before pyplot import
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from knn import KNNClassifier, make_two_class_blobs, make_decision_grid
from dct import make_demo_signal, dct1d, compress_1d


# ============================================================
# KNN Tab
# ============================================================

class KNNTab(ttk.Frame):
    """Interactive KNN decision-boundary visualisation."""

    COLORS = ["#4C72B0", "#DD8452"]   # class 0, class 1
    BG_COLORS = ["#AEC6E8", "#F4C89A"]

    def __init__(self, parent: ttk.Notebook):
        super().__init__(parent)
        self._build_data()
        self._build_ui()
        self._update_plot()

    # ------------------------------------------------------------------
    def _build_data(self):
        self._X, self._y = make_two_class_blobs(n_per_class=40, seed=7)

    # ------------------------------------------------------------------
    def _build_ui(self):
        # ---- controls row ----
        ctrl = ttk.Frame(self)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=10, pady=6)

        ttk.Label(ctrl, text="k (neighbours):").pack(side=tk.LEFT)
        self._k_var = tk.IntVar(value=3)
        self._k_slider = ttk.Scale(
            ctrl, from_=1, to=15, orient=tk.HORIZONTAL,
            variable=self._k_var, length=220,
            command=lambda _: self._update_plot(),
        )
        self._k_slider.pack(side=tk.LEFT, padx=6)
        self._k_label = ttk.Label(ctrl, text="k = 3", width=7)
        self._k_label.pack(side=tk.LEFT)

        self._acc_label = ttk.Label(ctrl, text="Accuracy: —")
        self._acc_label.pack(side=tk.LEFT, padx=16)

        # explanation text
        info = (
            "KNN classifies a new point by looking at its k nearest training "
            "samples and taking a majority vote of their labels.\n"
            "Small k → complex boundary (can overfit).  "
            "Large k → smoother boundary (may underfit)."
        )
        ttk.Label(self, text=info, wraplength=700, justify=tk.LEFT).pack(
            side=tk.TOP, padx=10, anchor=tk.W
        )

        # ---- matplotlib canvas ----
        self._fig, self._ax = plt.subplots(figsize=(7, 4.5))
        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

    # ------------------------------------------------------------------
    def _update_plot(self):
        k = max(1, int(self._k_var.get()))
        self._k_label.config(text=f"k = {k}")

        clf = KNNClassifier(k=k)
        clf.fit(self._X, self._y)

        margin = 0.5
        x_min, x_max = self._X[:, 0].min() - margin, self._X[:, 0].max() + margin
        y_min, y_max = self._X[:, 1].min() - margin, self._X[:, 1].max() + margin

        xx, yy, grid = make_decision_grid(x_min, x_max, y_min, y_max, resolution=0.12)
        Z = clf.predict(grid).reshape(xx.shape)

        # leave-one-out accuracy on training set
        acc = clf.score(self._X, self._y)
        self._acc_label.config(text=f"Train accuracy: {acc:.0%}")

        ax = self._ax
        ax.clear()
        ax.contourf(xx, yy, Z, alpha=0.35, levels=[-0.5, 0.5, 1.5],
                    colors=self.BG_COLORS)
        for cls, col in zip([0, 1], self.COLORS):
            mask = self._y == cls
            ax.scatter(
                self._X[mask, 0], self._X[mask, 1],
                c=col, edgecolors="k", linewidths=0.5,
                s=50, label=f"Class {cls}",
            )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"KNN Decision Boundary  (k = {k})", fontsize=12)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.legend(loc="upper left", fontsize=9)
        self._fig.tight_layout()
        self._canvas.draw()


# ============================================================
# DCT Tab
# ============================================================

class DCTTab(ttk.Frame):
    """Interactive 1-D DCT compression visualisation."""

    def __init__(self, parent: ttk.Notebook):
        super().__init__(parent)
        self._signal = make_demo_signal(N=64)
        self._coeffs = dct1d(self._signal)
        self._build_ui()
        self._update_plot()

    # ------------------------------------------------------------------
    def _build_ui(self):
        # ---- controls row ----
        ctrl = ttk.Frame(self)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=10, pady=6)

        N = len(self._signal)
        ttk.Label(ctrl, text="Coefficients kept:").pack(side=tk.LEFT)
        self._keep_var = tk.IntVar(value=N // 4)
        self._keep_slider = ttk.Scale(
            ctrl, from_=1, to=N, orient=tk.HORIZONTAL,
            variable=self._keep_var, length=220,
            command=lambda _: self._update_plot(),
        )
        self._keep_slider.pack(side=tk.LEFT, padx=6)
        self._keep_label = ttk.Label(ctrl, text="", width=12)
        self._keep_label.pack(side=tk.LEFT)

        self._mse_label = ttk.Label(ctrl, text="MSE: —")
        self._mse_label.pack(side=tk.LEFT, padx=16)

        info = (
            "The DCT expresses a signal as a weighted sum of cosine functions "
            "at different frequencies.  Low-index coefficients carry most of "
            "the energy (energy compaction), so we can discard the "
            "high-frequency coefficients with little visible change – the "
            "basis of JPEG image compression."
        )
        ttk.Label(self, text=info, wraplength=700, justify=tk.LEFT).pack(
            side=tk.TOP, padx=10, anchor=tk.W
        )

        # ---- matplotlib canvas  (2 sub-plots) ----
        self._fig, (self._ax_sig, self._ax_dct) = plt.subplots(
            1, 2, figsize=(9, 3.8)
        )
        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

    # ------------------------------------------------------------------
    def _update_plot(self):
        N = len(self._signal)
        keep = max(1, min(N, int(self._keep_var.get())))
        self._keep_label.config(text=f"{keep} / {N} ({keep / N:.0%})")

        reconstructed = compress_1d(self._signal, keep=keep)
        mse = float(np.mean((self._signal - reconstructed) ** 2))
        self._mse_label.config(text=f"MSE: {mse:.4f}")

        # ---- left plot: original vs reconstructed ----
        ax = self._ax_sig
        ax.clear()
        ax.plot(self._signal, label="Original", color="#4C72B0", linewidth=1.5)
        ax.plot(reconstructed, "--", label=f"Reconstructed ({keep} coeff.)",
                color="#DD8452", linewidth=1.5)
        ax.set_title("Signal Reconstruction", fontsize=11)
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Amplitude")
        ax.legend(fontsize=8)

        # ---- right plot: DCT coefficient spectrum ----
        ax2 = self._ax_dct
        ax2.clear()
        bar_colors = ["#4C72B0" if i < keep else "#CCCCCC" for i in range(N)]
        ax2.bar(np.arange(N), np.abs(self._coeffs), color=bar_colors, width=1.0)
        ax2.axvline(keep - 0.5, color="red", linestyle="--", linewidth=1, label="cutoff")
        ax2.set_title("DCT Coefficient Magnitudes", fontsize=11)
        ax2.set_xlabel("Coefficient index (frequency)")
        ax2.set_ylabel("|DCT coefficient|")
        ax2.legend(fontsize=8)

        self._fig.tight_layout()
        self._canvas.draw()


# ============================================================
# Main application
# ============================================================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("KNN & DCT – Interactive Learning Tool")
        self.geometry("800x620")
        self.resizable(True, True)

        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        knn_tab = KNNTab(notebook)
        notebook.add(knn_tab, text="  K-Nearest Neighbours (KNN)  ")

        dct_tab = DCTTab(notebook)
        notebook.add(dct_tab, text="  Discrete Cosine Transform (DCT)  ")

        # status bar
        status = ttk.Label(
            self,
            text="Drag the sliders to explore how each parameter affects the algorithm.",
            relief=tk.SUNKEN,
            anchor=tk.W,
        )
        status.pack(side=tk.BOTTOM, fill=tk.X)


if __name__ == "__main__":
    app = App()
    app.mainloop()
