from __future__ import annotations

import importlib.util
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from data_handler import load_csv
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.metrics import confusion_matrix


DISPLAY_NAMES = {
    "knn": "KNN",
    "decision_tree": "Decision Tree",
}

MODEL_COLORS = {
    "knn": "#2563eb",
    "decision_tree": "#dc2626",
}

METRICS = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1_score", "F1 Score"),
]


class ModelComparisonApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("ML Dashboard: KNN vs Decision Tree")
        self.root.geometry("1280x860")
        self.root.configure(bg="#eef2f7")

        self.dataset_path: Path | None = None
        self.X = None
        self.y = None
        self.results: dict[str, dict[str, object]] | None = None
        self.canvas: FigureCanvasTkAgg | None = None

        self._build_styles()
        self._build_layout()
        self._load_default_dataset()

    def _build_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("App.TFrame", background="#eef2f7")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure(
            "Title.TLabel",
            background="#eef2f7",
            foreground="#0f172a",
            font=("Helvetica", 20, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background="#eef2f7",
            foreground="#475569",
            font=("Helvetica", 11),
        )
        style.configure(
            "Section.TLabel",
            background="#ffffff",
            foreground="#0f172a",
            font=("Helvetica", 12, "bold"),
        )
        style.configure(
            "Info.TLabel",
            background="#ffffff",
            foreground="#334155",
            font=("Helvetica", 10),
        )
        style.configure("Primary.TButton", font=("Helvetica", 11, "bold"), padding=10)
        style.configure("Action.TButton", font=("Helvetica", 10, "bold"), padding=9)
        style.configure("Dashboard.TButton", font=("Helvetica", 11, "bold"), padding=12)

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=18, style="App.TFrame")
        container.pack(fill="both", expand=True)

        ttk.Label(container, text="Structured ML Dashboard", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            container,
            text=(
                "Load a dataset, train once, and inspect KNN, Decision Tree, or side-by-side comparison views "
                "without retraining."
            ),
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(4, 16))

        controls_card = ttk.Frame(container, padding=16, style="Card.TFrame")
        controls_card.pack(fill="x", pady=(0, 14))

        header_row = ttk.Frame(controls_card, style="Card.TFrame")
        header_row.pack(fill="x")
        ttk.Label(header_row, text="Dataset Controls", style="Section.TLabel").pack(side="left")
        self.dataset_status = ttk.Label(
            header_row,
            text="Dataset: Not loaded",
            style="Info.TLabel",
        )
        self.dataset_status.pack(side="right")

        buttons_row = ttk.Frame(controls_card, style="Card.TFrame")
        buttons_row.pack(fill="x", pady=(12, 0))
        ttk.Button(
            buttons_row,
            text="Load Dataset",
            command=self.load_dataset,
            style="Primary.TButton",
        ).pack(side="right", padx=(10, 0))
        ttk.Button(
            buttons_row,
            text="Train Models",
            command=self.train_models,
            style="Action.TButton",
        ).pack(side="right")

        dashboard_card = ttk.Frame(container, padding=16, style="Card.TFrame")
        dashboard_card.pack(fill="x", pady=(0, 14))
        ttk.Label(dashboard_card, text="Dashboard Views", style="Section.TLabel").pack(anchor="w")

        dashboard_buttons = ttk.Frame(dashboard_card, style="Card.TFrame")
        dashboard_buttons.pack(fill="x", pady=(12, 0))
        ttk.Button(
            dashboard_buttons,
            text="KNN",
            command=self.show_knn,
            style="Dashboard.TButton",
        ).pack(side="left", padx=(0, 10))
        ttk.Button(
            dashboard_buttons,
            text="Decision Tree",
            command=self.show_dct,
            style="Dashboard.TButton",
        ).pack(side="left", padx=(0, 10))
        ttk.Button(
            dashboard_buttons,
            text="Comparison",
            command=self.show_comparison,
            style="Dashboard.TButton",
        ).pack(side="left")

        content_row = ttk.Frame(container, style="App.TFrame")
        content_row.pack(fill="both", expand=True)

        left_panel = ttk.Frame(content_row, padding=16, style="Card.TFrame")
        left_panel.pack(side="left", fill="both", expand=False, padx=(0, 10))
        left_panel.configure(width=400)
        ttk.Label(left_panel, text="Metrics & Insights", style="Section.TLabel").pack(anchor="w")

        self.output_text = tk.Text(
            left_panel,
            wrap="word",
            width=44,
            font=("Courier New", 10),
            bg="#0f172a",
            fg="#e2e8f0",
            insertbackground="#e2e8f0",
            relief="flat",
            padx=12,
            pady=12,
        )
        self.output_text.pack(fill="both", expand=True, pady=(12, 0))

        right_panel = ttk.Frame(content_row, padding=16, style="Card.TFrame")
        right_panel.pack(side="left", fill="both", expand=True)
        ttk.Label(right_panel, text="Visualizations", style="Section.TLabel").pack(anchor="w")

        self.plot_frame = ttk.Frame(right_panel, style="Card.TFrame")
        self.plot_frame.pack(fill="both", expand=True, pady=(12, 0))

        self.clear_output()
        self.display_output("Ready. Load a CSV dataset, then train the models once to populate the dashboard.\n")

    def clear_output(self) -> None:
        self.output_text.delete("1.0", tk.END)

    def display_output(self, message: str) -> None:
        self.output_text.insert(tk.END, message)
        self.output_text.see(tk.END)

    def _load_default_dataset(self) -> None:
        default_dataset = Path(__file__).resolve().parent / "dataset" / "IRIS.csv"
        if default_dataset.exists():
            try:
                self._load_dataset_from_path(default_dataset)
                self.display_output("Default dataset loaded successfully. Click 'Train Models' to generate results.\n")
            except Exception as exc:  # noqa: BLE001
                self.clear_output()
                self.display_output(f"Default dataset could not be loaded: {exc}\n")

    def load_dataset(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select CSV Dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if file_path:
            self._load_dataset_from_path(Path(file_path))

    def _load_dataset_from_path(self, file_path: Path) -> None:
        try:
            X, y = load_csv(str(file_path))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Dataset Error", str(exc))
            return

        self.dataset_path = file_path
        self.X = X
        self.y = y
        self.results = None

        self.dataset_status.config(text=f"Dataset: {file_path.name}")
        self.clear_output()
        self.display_output(f"Loaded dataset: {file_path}\n")
        self.display_output(f"Samples: {len(self.X)}\n")
        self.display_output(f"Features per sample: {self.X.shape[1] if len(self.X) else 0}\n")
        self.display_output(f"Targets loaded: {len(self.y)}\n")
        self.display_output("Training results cleared. Use 'Train Models' to train once and unlock all dashboard views.\n")
        self._clear_plot()

    def _ensure_dataset_loaded(self) -> bool:
        if self.X is None or self.y is None:
            messagebox.showwarning("Dataset Not Loaded", "Please load a dataset before training or visualizing results.")
            return False
        return True

    def _ensure_results_ready(self) -> bool:
        if not self._ensure_dataset_loaded():
            return False
        if self.results is None:
            messagebox.showwarning("Models Not Trained", "Please train the models first before opening dashboard views.")
            return False
        return True

    def train_models(self) -> None:
        if not self._ensure_dataset_loaded():
            return

        if importlib.util.find_spec("sklearn") is None:
            messagebox.showerror(
                "Missing Dependency",
                "scikit-learn is required to train the models. Install it with: pip install scikit-learn",
            )
            return

        from model import train_and_evaluate

        self.results = train_and_evaluate(self.X, self.y)
        self.clear_output()
        dataset_name = str(self.dataset_path) if self.dataset_path else "In-memory dataset"
        self.display_output(f"Training complete for dataset: {dataset_name}\n")
        self.display_output("Training ran once and the dashboard will now reuse self.results for every view.\n\n")

        for model_key in ("knn", "decision_tree"):
            metrics = self.results[model_key]
            self.display_output(self._format_model_metrics(model_key, metrics))
            self.display_output("\n")

        self.display_output("Choose KNN, Decision Tree, or Comparison to view modular visualizations.\n")
        self.show_comparison()

    def show_knn(self) -> None:
        if not self._ensure_results_ready():
            return

        metrics = self.results["knn"]
        self.clear_output()
        self.display_output("KNN Dashboard\n")
        self.display_output("=" * 72 + "\n")
        self.display_output(self._format_model_metrics("knn", metrics))
        self._draw_figure(self._build_single_model_figure("knn", metrics))

    def show_dct(self) -> None:
        if not self._ensure_results_ready():
            return

        metrics = self.results["decision_tree"]
        self.clear_output()
        self.display_output("Decision Tree Dashboard\n")
        self.display_output("=" * 72 + "\n")
        self.display_output(self._format_model_metrics("decision_tree", metrics))
        self._draw_figure(self._build_single_model_figure("decision_tree", metrics))

    def show_comparison(self) -> None:
        if not self._ensure_results_ready():
            return

        best_model = max(
            self.results,
            key=lambda model_name: float(self.results[model_name]["accuracy"]),
        )

        self.clear_output()
        self.display_output("Model Comparison Dashboard\n")
        self.display_output("=" * 72 + "\n")
        self.display_output(self._format_model_metrics("knn", self.results["knn"]))
        self.display_output("\n")
        self.display_output(self._format_model_metrics("decision_tree", self.results["decision_tree"]))
        self.display_output("\n")
        self.display_output(f"Best model based on accuracy: {DISPLAY_NAMES[best_model]}\n")
        self._draw_figure(self._build_comparison_figure(best_model))

    def _format_model_metrics(self, model_key: str, metrics: dict[str, object]) -> str:
        lines = [f"{DISPLAY_NAMES[model_key]}"]
        for metric_key, metric_label in METRICS:
            lines.append(f"- {metric_label}: {float(metrics[metric_key]):.4f}")

        lines.append("- Confusion Matrix:")
        matrix = confusion_matrix(metrics["y_true"], metrics["y_pred"])
        for row in matrix:
            lines.append(f"  {list(row)}")
        return "\n".join(lines) + "\n"

    def _build_single_model_figure(self, model_key: str, metrics: dict[str, object]) -> Figure:
        figure = Figure(figsize=(9.5, 4.8), dpi=100)
        metric_axis = figure.add_subplot(1, 2, 1)
        matrix_axis = figure.add_subplot(1, 2, 2)

        values = [float(metrics[metric_key]) for metric_key, _ in METRICS]
        labels = [label for _, label in METRICS]
        bars = metric_axis.bar(labels, values, color=MODEL_COLORS[model_key])
        metric_axis.set_title(f"{DISPLAY_NAMES[model_key]} Metrics")
        metric_axis.set_ylabel("Score")
        metric_axis.set_ylim(0, 1.05)
        metric_axis.tick_params(axis="x", rotation=25)
        metric_axis.grid(axis="y", linestyle="--", alpha=0.3)
        for bar, value in zip(bars, values):
            metric_axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        matrix = confusion_matrix(metrics["y_true"], metrics["y_pred"])
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap=sns.light_palette(MODEL_COLORS[model_key], as_cmap=True),
            cbar=False,
            ax=matrix_axis,
        )
        matrix_axis.set_title(f"{DISPLAY_NAMES[model_key]} Confusion Matrix")
        matrix_axis.set_xlabel("Predicted Label")
        matrix_axis.set_ylabel("True Label")

        figure.tight_layout()
        return figure

    def _build_comparison_figure(self, best_model: str) -> Figure:
        figure = Figure(figsize=(12, 7.5), dpi=100)
        chart_axis = figure.add_subplot(2, 2, 1)
        knn_axis = figure.add_subplot(2, 2, 3)
        dct_axis = figure.add_subplot(2, 2, 4)
        summary_axis = figure.add_subplot(2, 2, 2)

        metric_labels = [label for _, label in METRICS]
        model_keys = ["knn", "decision_tree"]
        x_positions = range(len(model_keys))
        bar_width = 0.18

        for index, (metric_key, metric_label) in enumerate(METRICS):
            offsets = [position + (index - 1.5) * bar_width for position in x_positions]
            values = [float(self.results[model_key][metric_key]) for model_key in model_keys]
            chart_axis.bar(offsets, values, width=bar_width, label=metric_label)

        chart_axis.set_title("Model Metrics Comparison")
        chart_axis.set_xticks(list(x_positions))
        chart_axis.set_xticklabels([DISPLAY_NAMES[key] for key in model_keys])
        chart_axis.set_ylabel("Score")
        chart_axis.set_ylim(0, 1.05)
        chart_axis.grid(axis="y", linestyle="--", alpha=0.3)
        chart_axis.legend(fontsize=8)

        sns.heatmap(
            confusion_matrix(self.results["knn"]["y_true"], self.results["knn"]["y_pred"]),
            annot=True,
            fmt="d",
            cmap=sns.light_palette(MODEL_COLORS["knn"], as_cmap=True),
            cbar=False,
            ax=knn_axis,
        )
        knn_axis.set_title("KNN Confusion Matrix")
        knn_axis.set_xlabel("Predicted Label")
        knn_axis.set_ylabel("True Label")

        sns.heatmap(
            confusion_matrix(
                self.results["decision_tree"]["y_true"],
                self.results["decision_tree"]["y_pred"],
            ),
            annot=True,
            fmt="d",
            cmap=sns.light_palette(MODEL_COLORS["decision_tree"], as_cmap=True),
            cbar=False,
            ax=dct_axis,
        )
        dct_axis.set_title("Decision Tree Confusion Matrix")
        dct_axis.set_xlabel("Predicted Label")
        dct_axis.set_ylabel("True Label")

        summary_axis.axis("off")
        summary_lines = [
            "Dashboard Summary",
            "",
            f"Best model: {DISPLAY_NAMES[best_model]}",
            f"KNN Accuracy: {float(self.results['knn']['accuracy']):.4f}",
            f"Decision Tree Accuracy: {float(self.results['decision_tree']['accuracy']):.4f}",
            "",
            "Metrics compared:",
        ]
        summary_lines.extend(f"- {label}" for label in metric_labels)
        summary_axis.text(0, 1, "\n".join(summary_lines), ha="left", va="top", fontsize=11)
        summary_axis.set_title("Best Model Highlight")

        figure.tight_layout()
        return figure

    def _clear_plot(self) -> None:
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None

    def _draw_figure(self, figure: Figure) -> None:
        self._clear_plot()
        self.canvas = FigureCanvasTkAgg(figure, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)


def main() -> None:
    root = tk.Tk()
    ModelComparisonApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()



def run_app() -> None:
    main()
