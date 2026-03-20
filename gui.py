from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


class ModelComparisonApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("KNN vs Decision Tree Comparison")
        self.root.geometry("900x650")
        self.root.configure(bg="#f4f6f8")

        self.dataset_path: Path | None = None
        self.feature_names: list[str] = []
        self.X: list[list[float]] | None = None
        self.y: list[str] | None = None

        self._build_styles()
        self._build_layout()

    def _build_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Helvetica", 18, "bold"), background="#f4f6f8", foreground="#1f2933")
        style.configure("Info.TLabel", font=("Helvetica", 11), background="#f4f6f8", foreground="#52606d")
        style.configure("Primary.TButton", font=("Helvetica", 11, "bold"), padding=10)
        style.configure("Secondary.TButton", font=("Helvetica", 10), padding=8)
        style.configure("Card.TFrame", background="#ffffff")

    def _build_layout(self) -> None:
        main_frame = ttk.Frame(self.root, padding=20, style="Card.TFrame")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        ttk.Label(main_frame, text="Model Comparison Dashboard", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            main_frame,
            text="Load a CSV dataset, then train KNN and Decision Tree models for a quick comparison.",
            style="Info.TLabel",
        ).pack(anchor="w", pady=(6, 16))

        controls = ttk.Frame(main_frame, style="Card.TFrame")
        controls.pack(fill="x", pady=(0, 16))

        self.dataset_label = ttk.Label(
            controls,
            text="Dataset: Not loaded (default: dataset/IRIS.csv)",
            style="Info.TLabel",
        )
        self.dataset_label.pack(side="left", fill="x", expand=True)

        ttk.Button(controls, text="Load Dataset", command=self.load_dataset, style="Primary.TButton").pack(side="right", padx=(8, 0))
        ttk.Button(controls, text="Train Models", command=self.train_models, style="Secondary.TButton").pack(side="right")

        self.output_text = tk.Text(
            main_frame,
            wrap="word",
            font=("Courier New", 10),
            bg="#0f172a",
            fg="#e2e8f0",
            insertbackground="#e2e8f0",
            relief="flat",
            padx=12,
            pady=12,
        )
        self.output_text.pack(fill="both", expand=True)
        self.display_output("Ready. Load a dataset to begin.\n")

    def clear_output(self) -> None:
        self.output_text.delete("1.0", tk.END)

    def display_output(self, message: str) -> None:
        self.output_text.insert(tk.END, message)
        self.output_text.see(tk.END)

    def load_dataset(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select CSV Dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if file_path:
            self._load_dataset_from_path(Path(file_path))

    def _load_dataset_from_path(self, file_path: Path) -> None:
        try:
            with file_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.reader(csv_file))
        except OSError as exc:
            messagebox.showerror("Dataset Error", f"Unable to load dataset:\n{exc}")
            return

        if len(rows) < 2 or len(rows[0]) < 2:
            messagebox.showerror("Dataset Error", "Dataset must contain a header and at least one feature column plus one target column.")
            return

        header, data_rows = rows[0], rows[1:]
        try:
            self.X = [[float(value) for value in row[:-1]] for row in data_rows]
        except ValueError:
            messagebox.showerror("Dataset Error", "All feature columns must contain numeric values.")
            return

        self.y = [row[-1] for row in data_rows]
        self.feature_names = header[:-1]
        self.dataset_path = file_path

        self.dataset_label.config(text=f"Dataset: {file_path}")
        self.clear_output()
        self.display_output(f"Loaded dataset: {file_path}\n")
        self.display_output(f"Rows: {len(data_rows)}\n")
        self.display_output(f"Feature columns: {', '.join(self.feature_names)}\n")
        self.display_output(f"Feature shape: ({len(self.X)}, {len(self.feature_names)})\n")
        self.display_output(f"Target samples: {len(self.y)}\n")

    def train_models(self) -> None:
        if self.X is None or self.y is None:
            default_dataset = Path(__file__).resolve().parent / "dataset" / "IRIS.csv"
            if default_dataset.exists():
                self._load_dataset_from_path(default_dataset)
            else:
                messagebox.showwarning("No Dataset", "Please load a dataset before training models.")
                return

        if importlib.util.find_spec("sklearn") is None:
            messagebox.showerror(
                "Missing Dependency",
                "scikit-learn is required to train the models. Install it with: pip install scikit-learn",
            )
            return

        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier

        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42,
            stratify=self.y,
        )

        models = {
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
        }

        self.clear_output()
        dataset_name = str(self.dataset_path) if self.dataset_path else "In-memory dataset"
        self.display_output(f"Training models using: {dataset_name}\n")
        self.display_output(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}\n\n")

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions)

            self.display_output(f"{model_name}\n")
            self.display_output(f"Accuracy: {accuracy:.4f}\n")
            self.display_output("Classification Report:\n")
            self.display_output(f"{report}\n")
            self.display_output("-" * 70 + "\n")


def main() -> None:
    root = tk.Tk()
    ModelComparisonApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

def run_app():
    root = tk.Tk()
    ModelComparisonApp(root)
    root.mainloop()    