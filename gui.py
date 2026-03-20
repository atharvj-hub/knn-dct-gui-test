from __future__ import annotations

import importlib.util
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


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
        self.root.geometry("1380x920")
        self.root.configure(bg="#eef2f7")

        self.dataset_path: Path | None = None
        self.df: pd.DataFrame | None = None
        self.target_column: str | None = None
        self.feature_columns: list[str] = []
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.feature_widgets: dict[str, ttk.Widget] = {}
        self.target_encoder: LabelEncoder | None = None
        self.class_names: list[str] = []
        self.results: dict[str, dict[str, object]] | None = None
        self.canvas: FigureCanvasTkAgg | None = None

        self.target_var = tk.StringVar()
        self.prediction_var = tk.StringVar(value="Predicted value: --")

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
        style.configure(
            "Prediction.TLabel",
            background="#ffffff",
            foreground="#166534",
            font=("Helvetica", 11, "bold"),
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
                "Load any labeled CSV, choose the target column, train once, and inspect KNN, Decision Tree, "
                "or side-by-side comparison views without retraining."
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
        buttons_row.pack(fill="x", pady=(12, 6))
        ttk.Button(
            buttons_row,
            text="Load Dataset",
            command=self.load_dataset,
            style="Primary.TButton",
        ).pack(side="left")
        ttk.Button(
            buttons_row,
            text="Train Models",
            command=self.train_models,
            style="Action.TButton",
        ).pack(side="left", padx=(10, 0))

        target_row = ttk.Frame(controls_card, style="Card.TFrame")
        target_row.pack(fill="x", pady=(8, 0))
        ttk.Label(target_row, text="Select Target Column:", style="Info.TLabel").pack(side="left")
        self.target_dropdown = ttk.Combobox(
            target_row,
            textvariable=self.target_var,
            state="readonly",
            width=32,
        )
        self.target_dropdown.pack(side="left", padx=(12, 0))
        self.target_dropdown.bind("<<ComboboxSelected>>", self._on_target_selected)

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
        left_panel.configure(width=420)
        ttk.Label(left_panel, text="Metrics, Inputs & Prediction", style="Section.TLabel").pack(anchor="w")

        self.output_text = tk.Text(
            left_panel,
            wrap="word",
            width=48,
            height=18,
            font=("Courier New", 10),
            bg="#0f172a",
            fg="#e2e8f0",
            insertbackground="#e2e8f0",
            relief="flat",
            padx=12,
            pady=12,
        )
        self.output_text.pack(fill="x", pady=(12, 12))

        prediction_card = ttk.Frame(left_panel, style="Card.TFrame")
        prediction_card.pack(fill="both", expand=True)
        ttk.Label(prediction_card, text="Dynamic Input Fields", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            prediction_card,
            text="Numeric fields accept numbers. Categorical fields provide training-set values.",
            style="Info.TLabel",
        ).pack(anchor="w", pady=(4, 10))

        self.inputs_canvas = tk.Canvas(
            prediction_card,
            bg="#ffffff",
            highlightthickness=0,
            bd=0,
            relief="flat",
            height=260,
        )
        self.inputs_scrollbar = ttk.Scrollbar(
            prediction_card,
            orient="vertical",
            command=self.inputs_canvas.yview,
        )
        self.inputs_container = ttk.Frame(prediction_card, style="Card.TFrame")
        self.inputs_container.bind(
            "<Configure>",
            lambda event: self.inputs_canvas.configure(scrollregion=self.inputs_canvas.bbox("all")),
        )
        self.inputs_canvas.create_window((0, 0), window=self.inputs_container, anchor="nw")
        self.inputs_canvas.configure(yscrollcommand=self.inputs_scrollbar.set)
        self.inputs_canvas.pack(side="left", fill="both", expand=True)
        self.inputs_scrollbar.pack(side="right", fill="y")

        prediction_actions = ttk.Frame(left_panel, style="Card.TFrame")
        prediction_actions.pack(fill="x", pady=(12, 0))
        ttk.Button(
            prediction_actions,
            text="Predict",
            command=self.predict,
            style="Primary.TButton",
        ).pack(side="left")
        ttk.Label(
            prediction_actions,
            textvariable=self.prediction_var,
            style="Prediction.TLabel",
        ).pack(side="left", padx=(12, 0))

        right_panel = ttk.Frame(content_row, padding=16, style="Card.TFrame")
        right_panel.pack(side="left", fill="both", expand=True)
        ttk.Label(right_panel, text="Visualizations", style="Section.TLabel").pack(anchor="w")

        self.plot_frame = ttk.Frame(right_panel, style="Card.TFrame")
        self.plot_frame.pack(fill="both", expand=True, pady=(12, 0))

        self.clear_output()
        self.display_output("Ready. Load a CSV dataset, select a target column, then train the models.\n")

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
                self.display_output(
                    "Default dataset loaded successfully. Select a target column and click 'Train Models'.\n"
                )
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
            df = pd.read_csv(file_path)
        except FileNotFoundError as exc:
            messagebox.showerror("Dataset Error", f"CSV file not found: {file_path}")
            raise exc
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Dataset Error", f"Unable to load CSV file '{file_path}': {exc}")
            return

        if df.empty:
            messagebox.showerror("Dataset Error", "The selected CSV file is empty.")
            return

        self.dataset_path = file_path
        self.df = df.copy()
        self.results = None
        self.target_column = None
        self.feature_columns = []
        self.numeric_columns = []
        self.categorical_columns = []
        self.feature_widgets = {}
        self.target_encoder = None
        self.class_names = []
        self.target_var.set("")
        self.target_dropdown["values"] = df.columns.tolist()
        self.prediction_var.set("Predicted value: --")

        self.dataset_status.config(text=f"Dataset: {file_path.name}")
        self.clear_output()
        self.display_output(f"Loaded dataset: {file_path}\n")
        self.display_output(f"Rows: {len(df)}\n")
        self.display_output(f"Columns: {len(df.columns)}\n")
        self.display_output("Select the target column to configure features and prediction inputs.\n")
        self._rebuild_dynamic_inputs()
        self._clear_plot()

    def _on_target_selected(self, _event: tk.Event | None = None) -> None:
        if self.df is None:
            return

        self.target_column = self.target_var.get()
        if not self.target_column:
            return

        self.results = None
        self.prediction_var.set("Predicted value: --")
        self._configure_feature_metadata()
        self._rebuild_dynamic_inputs()

        self.clear_output()
        self.display_output(f"Loaded dataset: {self.dataset_path}\n")
        self.display_output(f"Target column selected: {self.target_column}\n")
        self.display_output(f"Feature columns ({len(self.feature_columns)}): {', '.join(self.feature_columns)}\n")
        self.display_output(
            f"Numeric features: {', '.join(self.numeric_columns) if self.numeric_columns else 'None'}\n"
        )
        self.display_output(
            f"Categorical features: {', '.join(self.categorical_columns) if self.categorical_columns else 'None'}\n"
        )
        self.display_output("Training results cleared. Click 'Train Models' to fit KNN and Decision Tree.\n")
        self._clear_plot()

    def _configure_feature_metadata(self) -> None:
        if self.df is None or self.target_column is None:
            self.feature_columns = []
            self.numeric_columns = []
            self.categorical_columns = []
            return

        self.feature_columns = [column for column in self.df.columns if column != self.target_column]
        feature_df = self.df[self.feature_columns].copy()
        self.numeric_columns = feature_df.select_dtypes(include="number").columns.tolist()
        self.categorical_columns = [
            column for column in self.feature_columns if column not in self.numeric_columns
        ]

    def _rebuild_dynamic_inputs(self) -> None:
        for widget in self.inputs_container.winfo_children():
            widget.destroy()

        self.feature_widgets = {}

        if not self.feature_columns:
            ttk.Label(
                self.inputs_container,
                text="Load a dataset and choose a target column to generate prediction fields.",
                style="Info.TLabel",
                wraplength=320,
            ).grid(row=0, column=0, sticky="w")
            return

        for row_index, column in enumerate(self.feature_columns):
            ttk.Label(
                self.inputs_container,
                text=f"{self._format_column_label(column)}:",
                style="Info.TLabel",
            ).grid(row=row_index, column=0, sticky="w", padx=(0, 12), pady=6)

            if column in self.categorical_columns:
                values = sorted(self.df[column].dropna().astype(str).unique().tolist()) if self.df is not None else []
                widget = ttk.Combobox(self.inputs_container, state="readonly", values=values, width=26)
                if values:
                    widget.set(values[0])
            else:
                widget = ttk.Entry(self.inputs_container, width=30)
                if self.df is not None:
                    numeric_series = pd.to_numeric(self.df[column], errors="coerce").dropna()
                    if not numeric_series.empty:
                        widget.insert(0, str(numeric_series.iloc[0]))

            widget.grid(row=row_index, column=1, sticky="ew", pady=6)
            self.feature_widgets[column] = widget

        self.inputs_container.columnconfigure(1, weight=1)

    def _format_column_label(self, column_name: str) -> str:
        return column_name.replace("_", " ").title()

    def _ensure_dataset_loaded(self) -> bool:
        if self.df is None:
            messagebox.showwarning("Dataset Not Loaded", "Please load a dataset before training or visualizing results.")
            return False
        return True

    def _ensure_target_selected(self) -> bool:
        if not self._ensure_dataset_loaded():
            return False
        if not self.target_column:
            messagebox.showwarning("Target Not Selected", "Please choose the target column before training or predicting.")
            return False
        return True

    def _ensure_results_ready(self) -> bool:
        if not self._ensure_target_selected():
            return False
        if self.results is None:
            messagebox.showwarning("Models Not Trained", "Please train the models first before opening dashboard views.")
            return False
        return True

    def train_models(self) -> None:
        if not self._ensure_target_selected():
            return

        if importlib.util.find_spec("sklearn") is None:
            messagebox.showerror(
                "Missing Dependency",
                "scikit-learn is required to train the models. Install it with: pip install scikit-learn",
            )
            return

        try:
            X, y = self._prepare_training_data()
            self.results = self._train_and_evaluate_models(X, y)
        except ValueError as exc:
            messagebox.showerror("Training Error", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Training Error", f"Unable to train models: {exc}")
            return

        self.clear_output()
        dataset_name = str(self.dataset_path) if self.dataset_path else "In-memory dataset"
        self.display_output(f"Training complete for dataset: {dataset_name}\n")
        self.display_output(f"Target column: {self.target_column}\n")
        self.display_output("Training ran once and the dashboard will now reuse self.results for every view.\n\n")

        for model_key in ("knn", "decision_tree"):
            metrics = self.results[model_key]
            self.display_output(self._format_model_metrics(model_key, metrics))
            self.display_output("\n")

        self.display_output("Choose KNN, Decision Tree, or Comparison to view modular visualizations.\n")
        self.show_comparison()

    def _prepare_training_data(self) -> tuple[pd.DataFrame, pd.Series]:
        if self.df is None or self.target_column is None:
            raise ValueError("A dataset and target column are required.")

        X = self.df.drop(columns=[self.target_column]).copy()
        y = self.df[self.target_column].copy()

        if X.empty:
            raise ValueError("The selected target column leaves no feature columns to train on.")

        if y.isna().any():
            raise ValueError("The selected target column contains missing values. Please clean the dataset first.")

        if y.dtype == object or pd.api.types.is_categorical_dtype(y) or pd.api.types.is_string_dtype(y):
            self.target_encoder = LabelEncoder()
            y = pd.Series(self.target_encoder.fit_transform(y.astype(str)), index=y.index, name=y.name)
            self.class_names = self.target_encoder.classes_.astype(str).tolist()
        else:
            self.target_encoder = None
            self.class_names = [str(label) for label in sorted(pd.Series(y).dropna().unique().tolist())]

        return X, y

    def _build_preprocessor(self) -> ColumnTransformer:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_columns),
                ("cat", categorical_transformer, self.categorical_columns),
            ],
            remainder="drop",
        )

    def _train_and_evaluate_models(self, X: pd.DataFrame, y: pd.Series) -> dict[str, dict[str, object]]:
        if y.nunique() < 2:
            raise ValueError("The selected target column must contain at least two classes.")

        stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )

        knn_pipeline = Pipeline(
            steps=[
                ("preprocessor", self._build_preprocessor()),
                ("scaler", StandardScaler(with_mean=False)),
                ("classifier", KNeighborsClassifier(n_neighbors=3)),
            ]
        )
        decision_tree_pipeline = Pipeline(
            steps=[
                ("preprocessor", self._build_preprocessor()),
                ("classifier", DecisionTreeClassifier(random_state=42)),
            ]
        )

        models = {
            "knn": knn_pipeline,
            "decision_tree": decision_tree_pipeline,
        }

        results: dict[str, dict[str, object]] = {}
        average_type = "binary" if y.nunique() == 2 else "macro"
        labels = sorted(pd.Series(y).unique().tolist())

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            results[model_name] = {
                "accuracy": float((predictions == y_test).mean()),
                "precision": self._safe_metric("precision", y_test, predictions, average_type),
                "recall": self._safe_metric("recall", y_test, predictions, average_type),
                "f1_score": self._safe_metric("f1", y_test, predictions, average_type),
                "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels),
                "y_true": y_test.to_numpy(),
                "y_pred": predictions,
                "model": model,
                "labels": labels,
            }

        return results

    def _safe_metric(self, metric_name: str, y_true: pd.Series, y_pred, average: str) -> float:
        from sklearn.metrics import f1_score, precision_score, recall_score

        metric_functions = {
            "precision": precision_score,
            "recall": recall_score,
            "f1": f1_score,
        }
        return float(metric_functions[metric_name](y_true, y_pred, average=average, zero_division=0))

    def predict(self) -> None:
        if not self._ensure_results_ready():
            return

        try:
            input_frame = self._collect_prediction_input()
            best_model_key = max(
                self.results,
                key=lambda model_name: float(self.results[model_name]["accuracy"]),
            )
            model = self.results[best_model_key]["model"]
            prediction = model.predict(input_frame)[0]
            predicted_value = self._decode_prediction(prediction)
        except ValueError as exc:
            messagebox.showerror("Prediction Error", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Prediction Error", f"Unable to generate prediction: {exc}")
            return

        self.prediction_var.set(f"Predicted value: {predicted_value} ({DISPLAY_NAMES[best_model_key]})")
        self.display_output(
            f"Prediction generated with {DISPLAY_NAMES[best_model_key]} using current form values: {predicted_value}\n"
        )

    def _collect_prediction_input(self) -> pd.DataFrame:
        if not self.feature_columns:
            raise ValueError("No feature columns are available for prediction.")

        row: dict[str, object] = {}
        for column in self.feature_columns:
            widget = self.feature_widgets.get(column)
            if widget is None:
                raise ValueError(f"Missing input widget for feature '{column}'.")

            raw_value = widget.get().strip()
            if raw_value == "":
                raise ValueError(f"Please provide a value for '{column}'.")

            if column in self.numeric_columns:
                try:
                    row[column] = float(raw_value)
                except ValueError as exc:
                    raise ValueError(f"Feature '{column}' requires a numeric value.") from exc
            else:
                row[column] = raw_value

        return pd.DataFrame([row], columns=self.feature_columns)

    def _decode_prediction(self, prediction: object) -> str:
        if self.target_encoder is not None:
            return str(self.target_encoder.inverse_transform([int(prediction)])[0])
        return str(prediction)

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
        matrix = metrics["confusion_matrix"]
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

        matrix = metrics["confusion_matrix"]
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap=sns.light_palette(MODEL_COLORS[model_key], as_cmap=True),
            cbar=False,
            ax=matrix_axis,
            xticklabels=self.class_names or "auto",
            yticklabels=self.class_names or "auto",
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
            self.results["knn"]["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap=sns.light_palette(MODEL_COLORS["knn"], as_cmap=True),
            cbar=False,
            ax=knn_axis,
            xticklabels=self.class_names or "auto",
            yticklabels=self.class_names or "auto",
        )
        knn_axis.set_title("KNN Confusion Matrix")
        knn_axis.set_xlabel("Predicted Label")
        knn_axis.set_ylabel("True Label")

        sns.heatmap(
            self.results["decision_tree"]["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap=sns.light_palette(MODEL_COLORS["decision_tree"], as_cmap=True),
            cbar=False,
            ax=dct_axis,
            xticklabels=self.class_names or "auto",
            yticklabels=self.class_names or "auto",
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
            f"Target column: {self.target_column}",
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
