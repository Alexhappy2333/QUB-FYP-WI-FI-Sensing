from __future__ import print_function

import json
import os
import random
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as scio
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Bidirectional,
    Concatenate,
    Conv1D,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GRU,
    Input,
    MaxPooling1D,
    MaxPooling2D,
    Reshape,
    TimeDistributed,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical


def env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value is not None else default


def env_float(name, default):
    value = os.environ.get(name)
    return float(value) if value is not None else default


def env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_path(name, default):
    value = os.environ.get(name)
    if value is None:
        return Path(default).resolve()
    return Path(value).expanduser().resolve()


# =========================================================
# 0. PATH CONFIG
# =========================================================
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = env_path(
    "TRAIN_DATA_DIR",
    r"F:\Wireless_Sensing\chen_CSI\Chen_new\experiments\ablation_runs\generated_datasets\dfs_no_column_norm",
)
OUTPUT_DIR = env_path("TRAIN_OUTPUT_DIR", ROOT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = OUTPUT_DIR / "best_model.keras"
TRAINING_HISTORY_PLOT_PATH = OUTPUT_DIR / "training_history.png"
CONFUSION_PLOT_PATH = OUTPUT_DIR / "confusion_matrix.png"
TEST_PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"
MODEL_CONFIG_PATH = OUTPUT_DIR / "model_config.json"
DATASET_SUMMARY_PATH = OUTPUT_DIR / "dataset_summary.json"
LENGTH_DISTRIBUTION_PLOT_PATH = OUTPUT_DIR / "length_distribution.png"
CLASS_MEAN_DFS_PLOT_PATH = OUTPUT_DIR / "class_mean_dfs.png"

CLASS_NAMES = ["clap", "push", "tap"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
N_CLASS = len(CLASS_NAMES)
PUSH_CLASS_ID = CLASS_TO_ID["push"]

# =========================================================
# 1. TRAIN CONFIG
# =========================================================
SEED = env_int("TRAIN_SEED", 2026)
TEST_SIZE = env_float("TRAIN_TEST_SIZE", 0.2)
VAL_SPLIT = env_float("TRAIN_VAL_SPLIT", 0.2)
N_EPOCHS = env_int("TRAIN_EPOCHS", 50)
BATCH_SIZE = env_int("TRAIN_BATCH_SIZE", 8)
LEARNING_RATE = env_float("TRAIN_LR", 1e-3)
DROPOUT_RATIO = env_float("TRAIN_DROPOUT", 0.35)
T_TARGET = env_int("TRAIN_T_TARGET", 24)
FILTER_LENGTH_OUTLIERS = env_bool("TRAIN_FILTER_LENGTH_OUTLIERS", False)
OUTLIER_IQR_FACTOR = env_float("TRAIN_OUTLIER_IQR_FACTOR", 1.5)
MIN_SAMPLES_FOR_OUTLIER_FILTER = env_int("TRAIN_MIN_SAMPLES_FOR_OUTLIER_FILTER", 8)

TRAIN_NORMALIZATION = os.environ.get("TRAIN_NORMALIZATION", "log_zscore").strip().lower()
TRAIN_LOSS = os.environ.get("TRAIN_LOSS", "cce").strip().lower()
TRAIN_MODEL_NAME = os.environ.get("TRAIN_MODEL_NAME", "tdconv_bigru").strip().lower()
TRAIN_USE_DELTA_CHANNEL = env_bool("TRAIN_USE_DELTA_CHANNEL", False)
TRAIN_USE_CLASS_WEIGHT = env_bool("TRAIN_USE_CLASS_WEIGHT", True)
TRAIN_USE_RAW_T_FEATURE = env_bool("TRAIN_USE_RAW_T_FEATURE", False)
FOCAL_GAMMA = env_float("TRAIN_FOCAL_GAMMA", 2.0)
FOCAL_ALPHA_MODE = os.environ.get("TRAIN_FOCAL_ALPHA_MODE", "auto_from_class_freq").strip().lower()
TRAIN_PUSH_WEIGHT_MULT = env_float("TRAIN_PUSH_WEIGHT_MULT", 1.4)
TRAIN_CLAP_WEIGHT_MULT = env_float("TRAIN_CLAP_WEIGHT_MULT", 1.0)
TRAIN_TAP_WEIGHT_MULT = env_float("TRAIN_TAP_WEIGHT_MULT", 1.0)

# =========================================================
# 2. RUNTIME STATE
# =========================================================
F_DIM = None
CHANNEL_DIM = None
DFS_RETURN_ONESIDED = None
DFS_COLUMN_NORMALIZE = None
MONITOR_METRIC = "val_macro_f1"
MONITOR_MODE = "max"


def configure_runtime():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Detected GPU devices:", [gpu.name for gpu in gpus])
        except RuntimeError as exc:
            print("GPU memory growth warning:", exc)


configure_runtime()


def summarize_lengths(lengths):
    arr = np.asarray(lengths, dtype=np.int32)
    if arr.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "unique_counts": {},
        }

    return {
        "count": int(arr.size),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "unique_counts": {
            str(length): int(count)
            for length, count in sorted(Counter(arr.tolist()).items())
        },
    }


def summarize_label_distribution(labels):
    counts = Counter(np.asarray(labels, dtype=np.int32).tolist())
    return {CLASS_NAMES[idx]: int(counts.get(idx, 0)) for idx in range(N_CLASS)}


def print_length_stats(title, lengths_by_class):
    print(f"\n{title}")
    for class_name in CLASS_NAMES:
        stats = summarize_lengths(lengths_by_class[class_name])
        if stats["count"] == 0:
            print(f"[{class_name}] no samples")
            continue

        print(f"\nClass: {class_name}")
        print(f"n = {stats['count']}")
        print(f"min_T = {stats['min']}")
        print(f"max_T = {stats['max']}")
        print(f"mean_T = {stats['mean']:.4f}")
        print(f"median_T = {stats['median']:.4f}")
        print(f"std_T = {stats['std']:.4f}")
        print(f"unique_T_counts = {stats['unique_counts']}")


def print_label_distribution(title, labels):
    print(f"{title} label distribution:", summarize_label_distribution(labels))


def build_length_filter(lengths_by_class):
    filter_bounds = {}

    for class_name in CLASS_NAMES:
        arr = np.asarray(lengths_by_class[class_name], dtype=np.float32)
        if arr.size < MIN_SAMPLES_FOR_OUTLIER_FILTER:
            filter_bounds[class_name] = None
            continue

        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lower = max(1.0, float(q1 - OUTLIER_IQR_FACTOR * iqr))
        upper = float(q3 + OUTLIER_IQR_FACTOR * iqr)
        filter_bounds[class_name] = {
            "lower": lower,
            "upper": upper,
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
        }

    return filter_bounds


def is_length_outlier(length_value, bounds):
    if bounds is None:
        return False
    return bool(length_value < bounds["lower"] or length_value > bounds["upper"])


def resample_time_axis(sample, target_t):
    sample = np.asarray(sample, dtype=np.float32)
    if sample.ndim == 2:
        freq_dim, time_dim = sample.shape
        channel_dim = None
    elif sample.ndim == 3:
        freq_dim, time_dim, channel_dim = sample.shape
    else:
        raise ValueError(f"sample must be 2D or 3D, got shape {sample.shape}")

    if time_dim == target_t:
        return sample

    if time_dim == 1:
        return np.repeat(sample, target_t, axis=1).astype(np.float32)

    old_t = np.linspace(0.0, 1.0, time_dim, dtype=np.float32)
    new_t = np.linspace(0.0, 1.0, target_t, dtype=np.float32)
    if channel_dim is None:
        out = np.empty((freq_dim, target_t), dtype=np.float32)
        for f_idx in range(freq_dim):
            out[f_idx] = np.interp(new_t, old_t, sample[f_idx]).astype(np.float32)
        return out

    out = np.empty((freq_dim, target_t, channel_dim), dtype=np.float32)
    for f_idx in range(freq_dim):
        for channel_idx in range(channel_dim):
            out[f_idx, :, channel_idx] = np.interp(
                new_t,
                old_t,
                sample[f_idx, :, channel_idx],
            ).astype(np.float32)

    return out


def transform_sample(sample, normalization_name):
    x = np.asarray(sample, dtype=np.float32)

    if normalization_name == "minmax":
        x = np.clip(x, a_min=0.0, a_max=None)
    elif normalization_name == "log_zscore":
        x = np.log1p(np.clip(x, a_min=0.0, a_max=None))
    else:
        raise ValueError("TRAIN_NORMALIZATION must be 'minmax' or 'log_zscore'.")

    x = resample_time_axis(x, T_TARGET)
    if x.ndim == 2:
        x = x.T
        x = np.expand_dims(x, axis=-1)
    elif x.ndim == 3:
        x = np.transpose(x, (1, 0, 2))
    else:
        raise ValueError(f"unsupported sample ndim after resample: {x.ndim}")
    return x.astype(np.float32)


def compute_normalization_stats(train_data, normalization_name):
    x = np.asarray(train_data, dtype=np.float32)

    if normalization_name == "minmax":
        return {
            "mode": "global_minmax",
            "min": float(np.min(x)),
            "max": float(np.max(x)),
        }

    if normalization_name == "log_zscore":
        return {
            "mode": "global_zscore",
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
        }

    raise ValueError("TRAIN_NORMALIZATION must be 'minmax' or 'log_zscore'.")


def apply_normalization(data, stats):
    x = np.asarray(data, dtype=np.float32)

    if stats["mode"] == "global_minmax":
        return (x - float(stats["min"])) / (float(stats["max"]) - float(stats["min"]) + 1e-8)

    if stats["mode"] == "global_zscore":
        return (x - float(stats["mean"])) / (float(stats["std"]) + 1e-6)

    raise ValueError(f"Unsupported normalization mode: {stats['mode']}")


def add_delta_channel(data, use_delta_channel):
    x = np.asarray(data, dtype=np.float32)
    if not use_delta_channel:
        return x

    delta = np.diff(x, axis=1, prepend=np.zeros_like(x[:, :1, :, :]))
    return np.concatenate([x, delta.astype(np.float32)], axis=-1)


def make_visualization_tensor(records):
    tensors = []
    labels = []

    for record in records:
        sample = np.log1p(np.clip(np.asarray(record["sample"], dtype=np.float32), a_min=0.0, a_max=None))
        sample = resample_time_axis(sample, T_TARGET)
        if sample.ndim == 2:
            sample = np.expand_dims(sample.T, axis=-1)
        elif sample.ndim == 3:
            sample = np.transpose(sample, (1, 0, 2))
        else:
            raise ValueError(f"unsupported visualization sample ndim: {sample.ndim}")
        tensors.append(sample)
        labels.append(record["label"])

    return np.asarray(tensors, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def prepare_split_data(records, indices, normalization_name, normalization_stats=None, use_delta_channel=False):
    base = np.asarray(
        [transform_sample(records[idx]["sample"], normalization_name) for idx in indices],
        dtype=np.float32,
    )

    if normalization_stats is None:
        normalization_stats = compute_normalization_stats(base, normalization_name)

    normalized = apply_normalization(base, normalization_stats)
    final_data = add_delta_channel(normalized, use_delta_channel)
    return final_data, normalization_stats


def compute_raw_t_stats(time_lengths):
    x = np.asarray(time_lengths, dtype=np.float32).reshape(-1)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
    }


def normalize_raw_t(time_lengths, stats):
    x = np.asarray(time_lengths, dtype=np.float32).reshape(-1, 1)
    return ((x - float(stats["mean"])) / (float(stats["std"]) + 1e-6)).astype(np.float32)


def prepare_raw_t_feature(records, indices, stats=None):
    raw_t = np.asarray([records[idx]["time_length"] for idx in indices], dtype=np.float32)
    if stats is None:
        stats = compute_raw_t_stats(raw_t)
    normalized = normalize_raw_t(raw_t, stats)
    return normalized, stats


def load_dataset(path_to_data):
    global F_DIM, CHANNEL_DIM, DFS_RETURN_ONESIDED, DFS_COLUMN_NORMALIZE
    F_DIM = None
    CHANNEL_DIM = None
    DFS_RETURN_ONESIDED = None
    DFS_COLUMN_NORMALIZE = None

    raw_records = []
    raw_lengths_by_class = {name: [] for name in CLASS_NAMES}

    print("Loading DFS data from:", path_to_data)

    for class_name in CLASS_NAMES:
        class_dir = path_to_data / class_name
        if not class_dir.is_dir():
            print("Skip missing directory:", class_dir)
            continue

        for fpath in sorted(class_dir.glob("*.mat")):
            try:
                mat_data = scio.loadmat(fpath)
                if "doppler_spectrum" not in mat_data:
                    print("Skip file without doppler_spectrum:", fpath)
                    continue

                x = np.asarray(mat_data["doppler_spectrum"], dtype=np.float32)
                x = np.squeeze(x)
                if x.ndim not in {2, 3}:
                    print("Skip invalid shape:", fpath, x.shape)
                    continue

                if not np.all(np.isfinite(x)):
                    print("Skip non-finite sample:", fpath)
                    continue

                current_return_onesided = None
                current_column_normalize = None
                if "dfs_return_onesided" in mat_data:
                    current_return_onesided = bool(np.squeeze(mat_data["dfs_return_onesided"]).item())
                if "dfs_column_normalize" in mat_data:
                    current_column_normalize = bool(np.squeeze(mat_data["dfs_column_normalize"]).item())

                current_f_dim = int(x.shape[0])
                current_channel_dim = 1 if x.ndim == 2 else int(x.shape[2])
                if F_DIM is None:
                    F_DIM = current_f_dim
                    CHANNEL_DIM = current_channel_dim
                    DFS_RETURN_ONESIDED = current_return_onesided
                    DFS_COLUMN_NORMALIZE = current_column_normalize
                elif F_DIM != current_f_dim or CHANNEL_DIM != current_channel_dim:
                    print("Skip inconsistent DFS shape:", fpath, x.shape)
                    continue

                time_length = int(x.shape[1])
                raw_lengths_by_class[class_name].append(time_length)
                raw_records.append(
                    {
                        "class_name": class_name,
                        "label": CLASS_TO_ID[class_name],
                        "file_path": str(fpath),
                        "sample": x,
                        "time_length": time_length,
                    }
                )
            except Exception as exc:
                print("Failed to load:", fpath, "error:", exc)

    if not raw_records:
        raise RuntimeError("No valid samples were loaded from the DFS directory.")

    print_length_stats("Raw DFS time-length stats", raw_lengths_by_class)

    filter_bounds = build_length_filter(raw_lengths_by_class)
    removed_records = []
    kept_records = []

    for record in raw_records:
        bounds = filter_bounds[record["class_name"]]
        if FILTER_LENGTH_OUTLIERS and is_length_outlier(record["time_length"], bounds):
            removed_records.append(
                {
                    "class_name": record["class_name"],
                    "file_path": record["file_path"],
                    "time_length": record["time_length"],
                }
            )
            continue
        kept_records.append(record)

    if FILTER_LENGTH_OUTLIERS:
        print("\nLength outlier filter enabled with IQR factor =", OUTLIER_IQR_FACTOR)
        if removed_records:
            for record in removed_records:
                print(
                    f"Removed {record['class_name']} sample "
                    f"T={record['time_length']} -> {record['file_path']}"
                )
        else:
            print("No samples removed by the length filter.")
    else:
        print("\nLength outlier filter disabled.")

    if not kept_records:
        raise RuntimeError("All samples were removed before training.")

    used_lengths_by_class = {name: [] for name in CLASS_NAMES}
    labels = []
    file_paths = []

    for record in kept_records:
        used_lengths_by_class[record["class_name"]].append(record["time_length"])
        labels.append(record["label"])
        file_paths.append(record["file_path"])

    print_length_stats("Raw DFS time-length stats used for training", used_lengths_by_class)

    labels = np.asarray(labels, dtype=np.int32)
    file_paths = np.asarray(file_paths, dtype=object)

    dataset_summary = {
        "data_dir": str(path_to_data),
        "output_dir": str(OUTPUT_DIR),
        "class_names": CLASS_NAMES,
        "t_target": T_TARGET,
        "f_dim": F_DIM,
        "channel_dim": CHANNEL_DIM,
        "dfs_return_onesided": DFS_RETURN_ONESIDED,
        "dfs_column_normalize": DFS_COLUMN_NORMALIZE,
        "raw_sample_count": int(len(raw_records)),
        "used_sample_count": int(len(kept_records)),
        "filter_length_outliers": FILTER_LENGTH_OUTLIERS,
        "outlier_iqr_factor": OUTLIER_IQR_FACTOR,
        "raw_class_counts": {
            class_name: int(len(raw_lengths_by_class[class_name]))
            for class_name in CLASS_NAMES
        },
        "used_class_counts": {
            class_name: int(len(used_lengths_by_class[class_name]))
            for class_name in CLASS_NAMES
        },
        "raw_time_length_stats": {
            class_name: summarize_lengths(raw_lengths_by_class[class_name])
            for class_name in CLASS_NAMES
        },
        "used_time_length_stats": {
            class_name: summarize_lengths(used_lengths_by_class[class_name])
            for class_name in CLASS_NAMES
        },
        "length_filter_bounds": filter_bounds,
        "removed_samples": removed_records,
    }

    visualization_data, visualization_labels = make_visualization_tensor(kept_records)
    return kept_records, labels, file_paths, dataset_summary, visualization_data, visualization_labels


def plot_training_history(history_dict, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes = axes.ravel()

    plots = [
        ("accuracy", "val_accuracy", "Accuracy"),
        ("macro_f1", "val_macro_f1", "Macro-F1"),
        ("push_recall", "val_push_recall", "Push Recall"),
        ("loss", "val_loss", "Loss"),
    ]

    for axis, (train_key, val_key, title) in zip(axes, plots):
        train_values = history_dict.get(train_key, [])
        val_values = history_dict.get(val_key, [])
        epochs = range(1, len(train_values) + 1)

        if train_values:
            axis.plot(epochs, train_values, label=train_key)
        if val_values:
            axis.plot(epochs, val_values, label=val_key)

        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)
        if train_values or val_values:
            axis.legend()

    fig.suptitle("Training History")
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print("Saved history plot to:", save_path)


def plot_confusion_matrix_figure(cm, class_names, save_path, held_out_accuracy=None, macro_f1=None):
    cm_norm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4.5),
        gridspec_kw={"width_ratios": [1.0, 1.0, 0.72]},
        constrained_layout=True,
    )
    plots = [
        (cm, "Confusion Matrix (Count)", "d"),
        (cm_norm, "Confusion Matrix (Normalized)", ".2f"),
    ]

    for axis, (matrix, subplot_title, fmt) in zip(axes[:2], plots):
        im = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
        axis.set_title(subplot_title)
        axis.set_xticks(np.arange(len(class_names)))
        axis.set_yticks(np.arange(len(class_names)))
        axis.set_xticklabels(class_names, rotation=45)
        axis.set_yticklabels(class_names)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")

        threshold = float(np.max(matrix)) / 2.0 if matrix.size else 0.0
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = format(matrix[row_idx, col_idx], fmt)
                axis.text(
                    col_idx,
                    row_idx,
                    value,
                    ha="center",
                    va="center",
                    color="white" if matrix[row_idx, col_idx] > threshold else "black",
                )

        fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)

    summary_axis = axes[2]
    summary_axis.axis("off")
    summary_axis.set_title("Held-out Test Summary")

    if held_out_accuracy is not None:
        summary_axis.text(
            0.5,
            0.72,
            "Held-out test accuracy",
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
        )
        summary_axis.text(
            0.5,
            0.50,
            f"{held_out_accuracy:.2%}",
            ha="center",
            va="center",
            fontsize=34,
            fontweight="bold",
            color="#1F4E79",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#EAF2FB", "edgecolor": "#1F4E79"},
        )

    if macro_f1 is not None:
        summary_axis.text(
            0.5,
            0.24,
            "Macro-F1",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
        )
        summary_axis.text(
            0.5,
            0.11,
            f"{macro_f1:.4f}",
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
            color="#3B6C3B",
        )

    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print("Saved confusion matrix plot to:", save_path)


def plot_length_distribution(lengths_by_class, save_path):
    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(12, 3.5), sharey=True, constrained_layout=True)
    if len(CLASS_NAMES) == 1:
        axes = [axes]

    for axis, class_name in zip(axes, CLASS_NAMES):
        lengths = np.asarray(lengths_by_class[class_name], dtype=np.int32)
        if lengths.size == 0:
            axis.set_title(f"{class_name}\nno samples")
            continue

        bins = np.arange(lengths.min(), lengths.max() + 2) - 0.5
        axis.hist(lengths, bins=bins, color="#4C78A8", edgecolor="black", alpha=0.85)
        axis.set_title(class_name)
        axis.set_xlabel("Raw T")
        axis.grid(True, axis="y", alpha=0.25)

    axes[0].set_ylabel("Count")
    fig.suptitle("Per-class Raw Time Length Distribution")
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print("Saved length distribution plot to:", save_path)


def plot_class_mean_dfs(data, labels, save_path):
    mean_maps = []
    for class_idx in range(N_CLASS):
        class_samples = data[labels == class_idx]
        if class_samples.size == 0:
            mean_maps.append(None)
            continue
        mean_tensor = np.mean(class_samples, axis=0)
        mean_maps.append(np.mean(mean_tensor, axis=-1).T)

    valid_maps = [mean_map for mean_map in mean_maps if mean_map is not None]
    vmin = min(float(np.min(mean_map)) for mean_map in valid_maps) if valid_maps else 0.0
    vmax = max(float(np.max(mean_map)) for mean_map in valid_maps) if valid_maps else 1.0

    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(12, 4), sharey=True, constrained_layout=True)
    if len(CLASS_NAMES) == 1:
        axes = [axes]

    im = None
    for axis, class_name, mean_map in zip(axes, CLASS_NAMES, mean_maps):
        if mean_map is None:
            axis.set_title(f"{class_name}\nno samples")
            continue
        im = axis.imshow(mean_map, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        axis.set_title(class_name)
        axis.set_xlabel("Resampled Time")

    axes[0].set_ylabel("Frequency Bin")
    fig.suptitle("Per-class Mean DFS After Resample")
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print("Saved class mean DFS plot to:", save_path)


def build_tdconv_bigru_backbone(model_input):
    x = TimeDistributed(
        Conv1D(16, kernel_size=3, padding="same", activation="relu"),
        name="td_conv1",
    )(model_input)
    x = TimeDistributed(MaxPooling1D(pool_size=2), name="td_pool1")(x)

    x = TimeDistributed(
        Conv1D(32, kernel_size=3, padding="same", activation="relu"),
        name="td_conv2",
    )(x)

    x = TimeDistributed(Flatten(), name="td_flatten")(x)
    x = TimeDistributed(Dense(64, activation="relu"), name="td_dense")(x)
    x = BatchNormalization(name="batch_norm")(x)
    x = TimeDistributed(Dropout(DROPOUT_RATIO), name="td_dropout")(x)
    x = Bidirectional(GRU(64, return_sequences=False), name="bigru")(x)
    x = Dropout(DROPOUT_RATIO, name="final_dropout")(x)
    return x


def build_cnn2d_backbone(model_input):
    x = Conv2D(16, kernel_size=(3, 3), padding="same", activation="relu", name="conv2d_1")(model_input)
    x = BatchNormalization(name="conv2d_bn1")(x)
    x = MaxPooling2D(pool_size=(2, 2), name="pool2d_1")(x)
    x = Dropout(DROPOUT_RATIO, name="conv2d_dropout1")(x)

    x = Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu", name="conv2d_2")(x)
    x = BatchNormalization(name="conv2d_bn2")(x)
    x = MaxPooling2D(pool_size=(2, 1), name="pool2d_2")(x)
    x = Flatten(name="flatten")(x)
    x = Dense(64, activation="relu", name="dense_head")(x)
    x = Dropout(DROPOUT_RATIO, name="final_dropout")(x)
    return x


def build_cnn_gru_backbone(model_input, input_shape):
    time_steps = input_shape[0]

    x = Conv2D(16, kernel_size=(3, 3), padding="same", activation="relu", name="conv2d_1")(model_input)
    x = BatchNormalization(name="conv2d_bn1")(x)
    x = MaxPooling2D(pool_size=(1, 2), name="pool2d_1")(x)
    x = Dropout(DROPOUT_RATIO, name="conv2d_dropout1")(x)

    x = Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu", name="conv2d_2")(x)
    x = BatchNormalization(name="conv2d_bn2")(x)
    x = Reshape((time_steps, -1), name="cnn_to_sequence")(x)
    x = Bidirectional(GRU(64, return_sequences=False), name="bigru")(x)
    x = Dropout(DROPOUT_RATIO, name="final_dropout")(x)
    x = Dense(64, activation="relu", name="dense_head")(x)
    return x


def assemble_model(input_shape, n_class):
    model_input = Input(shape=input_shape, dtype="float32", name="model_input")

    if TRAIN_MODEL_NAME == "tdconv_bigru":
        x = build_tdconv_bigru_backbone(model_input)
    elif TRAIN_MODEL_NAME == "cnn2d":
        x = build_cnn2d_backbone(model_input)
    elif TRAIN_MODEL_NAME == "cnn_gru":
        x = build_cnn_gru_backbone(model_input, input_shape)
    else:
        raise ValueError("TRAIN_MODEL_NAME must be 'tdconv_bigru', 'cnn2d', or 'cnn_gru'.")

    model_inputs = [model_input]
    if TRAIN_USE_RAW_T_FEATURE:
        raw_t_input = Input(shape=(1,), dtype="float32", name="raw_t_input")
        raw_t_feature = Dense(8, activation="relu", name="raw_t_dense")(raw_t_input)
        x = Concatenate(name="feature_fusion")([x, raw_t_feature])
        model_inputs.append(raw_t_input)

    model_output = Dense(n_class, activation="softmax", name="model_output")(x)
    return Model(inputs=model_inputs, outputs=model_output, name=f"dfs_{TRAIN_MODEL_NAME}")


def compute_class_weight(labels):
    labels = np.asarray(labels, dtype=np.int32)
    counts = np.bincount(labels, minlength=N_CLASS).astype(np.float32)
    total = float(np.sum(counts))
    weights = total / (float(N_CLASS) * np.maximum(counts, 1.0))
    class_weight = {class_idx: float(weights[class_idx]) for class_idx in range(N_CLASS)}
    class_weight[CLASS_TO_ID["clap"]] *= TRAIN_CLAP_WEIGHT_MULT
    class_weight[CLASS_TO_ID["push"]] *= TRAIN_PUSH_WEIGHT_MULT
    class_weight[CLASS_TO_ID["tap"]] *= TRAIN_TAP_WEIGHT_MULT
    return class_weight


def compute_focal_alpha(labels):
    labels = np.asarray(labels, dtype=np.int32)
    counts = np.bincount(labels, minlength=N_CLASS).astype(np.float32)
    total = float(np.sum(counts))
    inverse = total / (float(N_CLASS) * np.maximum(counts, 1.0))
    inverse = inverse / float(np.mean(inverse))
    return inverse.astype(np.float32)


def make_loss(loss_name, train_labels):
    if loss_name == "cce":
        return "categorical_crossentropy", None

    if loss_name == "focal":
        alpha = None
        if FOCAL_ALPHA_MODE == "auto_from_class_freq":
            alpha = compute_focal_alpha(train_labels).tolist()
        elif FOCAL_ALPHA_MODE in {"none", ""}:
            alpha = None
        else:
            raise ValueError("TRAIN_FOCAL_ALPHA_MODE must be 'none' or 'auto_from_class_freq'.")

        return (
            tf.keras.losses.CategoricalFocalCrossentropy(
                alpha=alpha,
                gamma=FOCAL_GAMMA,
            ),
            alpha,
        )

    raise ValueError("TRAIN_LOSS must be 'cce' or 'focal'.")


def compile_model(model, train_labels):
    loss_obj, focal_alpha = make_loss(TRAIN_LOSS, train_labels)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss_obj,
        metrics=[
            "accuracy",
            tf.keras.metrics.F1Score(average="macro", threshold=None, name="macro_f1"),
            tf.keras.metrics.Recall(class_id=PUSH_CLASS_ID, name="push_recall"),
            tf.keras.metrics.Precision(class_id=PUSH_CLASS_ID, name="push_precision"),
        ],
    )
    return focal_alpha


class ValidationMetricsCallback(Callback):
    def __init__(self, validation_data, validation_labels):
        super().__init__()
        if isinstance(validation_data, (list, tuple)):
            self.validation_data = [np.asarray(item, dtype=np.float32) for item in validation_data]
        else:
            self.validation_data = np.asarray(validation_data, dtype=np.float32)
        self.validation_labels = np.asarray(validation_labels, dtype=np.int32)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_prob = self.model.predict(self.validation_data, batch_size=BATCH_SIZE, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)

        macro_f1 = float(f1_score(self.validation_labels, y_pred, average="macro", zero_division=0))
        push_recall = float(
            recall_score(
                self.validation_labels,
                y_pred,
                labels=[PUSH_CLASS_ID],
                average=None,
                zero_division=0,
            )[0]
        )
        push_precision = float(
            precision_score(
                self.validation_labels,
                y_pred,
                labels=[PUSH_CLASS_ID],
                average=None,
                zero_division=0,
            )[0]
        )

        logs["val_macro_f1"] = macro_f1
        logs["val_push_recall"] = push_recall
        logs["val_push_precision"] = push_precision

        print(
            f" - val_macro_f1: {macro_f1:.4f}"
            f" - val_push_recall: {push_recall:.4f}"
            f" - val_push_precision: {push_precision:.4f}"
        )


def history_to_float_dict(history):
    return {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }


def validate_runtime_config():
    if TRAIN_NORMALIZATION not in {"minmax", "log_zscore"}:
        raise ValueError("TRAIN_NORMALIZATION must be 'minmax' or 'log_zscore'.")
    if TRAIN_LOSS not in {"cce", "focal"}:
        raise ValueError("TRAIN_LOSS must be 'cce' or 'focal'.")
    if FOCAL_ALPHA_MODE not in {"none", "auto_from_class_freq"}:
        raise ValueError("TRAIN_FOCAL_ALPHA_MODE must be 'none' or 'auto_from_class_freq'.")
    if TRAIN_MODEL_NAME not in {"tdconv_bigru", "cnn2d", "cnn_gru"}:
        raise ValueError("TRAIN_MODEL_NAME must be 'tdconv_bigru', 'cnn2d', or 'cnn_gru'.")


def print_runtime_config():
    print("Current runtime config")
    print("model_name =", TRAIN_MODEL_NAME)
    print("normalization =", TRAIN_NORMALIZATION)
    print("loss =", TRAIN_LOSS)
    print("use_delta_channel =", TRAIN_USE_DELTA_CHANNEL)
    print("use_class_weight =", TRAIN_USE_CLASS_WEIGHT)
    print("use_raw_t_feature =", TRAIN_USE_RAW_T_FEATURE)
    print("t_target =", T_TARGET)
    print("dropout_ratio =", DROPOUT_RATIO)
    print("learning_rate =", LEARNING_RATE)
    print("focal_gamma =", FOCAL_GAMMA)
    print("focal_alpha_mode =", FOCAL_ALPHA_MODE)
    print("train_clap_weight_mult =", TRAIN_CLAP_WEIGHT_MULT)
    print("train_push_weight_mult =", TRAIN_PUSH_WEIGHT_MULT)
    print("train_tap_weight_mult =", TRAIN_TAP_WEIGHT_MULT)
    print("monitor metric =", MONITOR_METRIC)


def main():
    validate_runtime_config()

    if not (0.0 < TEST_SIZE < 0.5):
        raise ValueError("TRAIN_TEST_SIZE must be between 0 and 0.5.")
    if not (0.0 < VAL_SPLIT < 0.5):
        raise ValueError("TRAIN_VAL_SPLIT must be between 0 and 0.5.")

    print("ROOT_DIR =", ROOT_DIR)
    print("DATA_DIR =", DATA_DIR)
    print("OUTPUT_DIR =", OUTPUT_DIR)
    print_runtime_config()

    records, labels, file_paths, dataset_summary, visualization_data, visualization_labels = load_dataset(DATA_DIR)

    print("\nLoaded samples:", len(labels))
    print("F_DIM =", F_DIM, "CHANNEL_DIM =", CHANNEL_DIM, "T_TARGET =", T_TARGET)
    print_label_distribution("Full dataset", labels)

    raw_lengths_by_class = {
        class_name: [
            int(record_length)
            for record_length in dataset_summary["used_time_length_stats"][class_name]["unique_counts"].keys()
            for _ in range(dataset_summary["used_time_length_stats"][class_name]["unique_counts"][record_length])
        ]
        for class_name in CLASS_NAMES
    }
    plot_length_distribution(raw_lengths_by_class, LENGTH_DISTRIBUTION_PLOT_PATH)
    plot_class_mean_dfs(visualization_data, visualization_labels, CLASS_MEAN_DFS_PLOT_PATH)

    all_indices = np.arange(len(labels))
    train_val_idx, test_idx = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=labels,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=VAL_SPLIT,
        random_state=SEED,
        stratify=labels[train_val_idx],
    )

    label_train = labels[train_idx]
    label_val = labels[val_idx]
    label_test = labels[test_idx]

    print("\nDataset split")
    print("train =", len(train_idx))
    print("val   =", len(val_idx))
    print("test  =", len(test_idx))
    print_label_distribution("Train", label_train)
    print_label_distribution("Val", label_val)
    print_label_distribution("Test", label_test)

    dataset_summary["split"] = {
        "test_size": TEST_SIZE,
        "validation_split_on_trainval": VAL_SPLIT,
        "train_count": int(len(train_idx)),
        "val_count": int(len(val_idx)),
        "test_count": int(len(test_idx)),
        "train_class_counts": summarize_label_distribution(label_train),
        "val_class_counts": summarize_label_distribution(label_val),
        "test_class_counts": summarize_label_distribution(label_test),
    }

    x_train, normalization_stats = prepare_split_data(
        records=records,
        indices=train_idx,
        normalization_name=TRAIN_NORMALIZATION,
        normalization_stats=None,
        use_delta_channel=TRAIN_USE_DELTA_CHANNEL,
    )
    x_val, _ = prepare_split_data(
        records=records,
        indices=val_idx,
        normalization_name=TRAIN_NORMALIZATION,
        normalization_stats=normalization_stats,
        use_delta_channel=TRAIN_USE_DELTA_CHANNEL,
    )
    x_test, _ = prepare_split_data(
        records=records,
        indices=test_idx,
        normalization_name=TRAIN_NORMALIZATION,
        normalization_stats=normalization_stats,
        use_delta_channel=TRAIN_USE_DELTA_CHANNEL,
    )
    raw_t_train, raw_t_stats = prepare_raw_t_feature(records=records, indices=train_idx, stats=None)
    raw_t_val, _ = prepare_raw_t_feature(records=records, indices=val_idx, stats=raw_t_stats)
    raw_t_test, _ = prepare_raw_t_feature(records=records, indices=test_idx, stats=raw_t_stats)

    dataset_summary["current_config"] = {
        "normalization": TRAIN_NORMALIZATION,
        "loss": TRAIN_LOSS,
        "use_delta_channel": TRAIN_USE_DELTA_CHANNEL,
        "use_class_weight": TRAIN_USE_CLASS_WEIGHT,
        "use_raw_t_feature": TRAIN_USE_RAW_T_FEATURE,
        "dfs_return_onesided": DFS_RETURN_ONESIDED,
        "dfs_column_normalize": DFS_COLUMN_NORMALIZE,
        "focal_gamma": FOCAL_GAMMA,
        "focal_alpha_mode": FOCAL_ALPHA_MODE,
        "train_clap_weight_mult": TRAIN_CLAP_WEIGHT_MULT,
        "train_push_weight_mult": TRAIN_PUSH_WEIGHT_MULT,
        "train_tap_weight_mult": TRAIN_TAP_WEIGHT_MULT,
        "monitor_metric": MONITOR_METRIC,
        "normalization_stats": normalization_stats,
        "raw_t_stats": raw_t_stats,
    }
    DATASET_SUMMARY_PATH.write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")
    print("Saved dataset summary to:", DATASET_SUMMARY_PATH)

    dataset_summary["current_config"]["model_name"] = TRAIN_MODEL_NAME
    DATASET_SUMMARY_PATH.write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")
    print("Updated dataset summary with model config:", DATASET_SUMMARY_PATH)

    y_train_oh = to_categorical(label_train, num_classes=N_CLASS)
    y_val_oh = to_categorical(label_val, num_classes=N_CLASS)
    y_test_oh = to_categorical(label_test, num_classes=N_CLASS)

    input_shape = tuple(x_train.shape[1:])
    model = assemble_model(input_shape=input_shape, n_class=N_CLASS)
    focal_alpha = compile_model(model, label_train)

    class_weight = compute_class_weight(label_train) if TRAIN_USE_CLASS_WEIGHT else None
    print("class_weight =", class_weight)
    if focal_alpha is not None:
        print("focal alpha =", focal_alpha)

    fit_train_input = [x_train, raw_t_train] if TRAIN_USE_RAW_T_FEATURE else x_train
    fit_val_input = [x_val, raw_t_val] if TRAIN_USE_RAW_T_FEATURE else x_val
    fit_test_input = [x_test, raw_t_test] if TRAIN_USE_RAW_T_FEATURE else x_test
    metric_callback = ValidationMetricsCallback(validation_data=fit_val_input, validation_labels=label_val)
    checkpoint = ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH),
        monitor=MONITOR_METRIC,
        mode=MONITOR_MODE,
        save_best_only=True,
        verbose=1,
    )
    early_stop = EarlyStopping(
        monitor=MONITOR_METRIC,
        mode=MONITOR_MODE,
        patience=8,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor=MONITOR_METRIC,
        mode=MONITOR_MODE,
        factor=0.5,
        patience=4,
        min_lr=1e-5,
        verbose=1,
    )

    history = model.fit(
        fit_train_input,
        y_train_oh,
        batch_size=BATCH_SIZE,
        epochs=N_EPOCHS,
        verbose=1,
        validation_data=(fit_val_input, y_val_oh),
        shuffle=True,
        callbacks=[metric_callback, checkpoint, early_stop, reduce_lr],
        class_weight=class_weight,
    )

    history_dict = history_to_float_dict(history)
    plot_training_history(history_dict, TRAINING_HISTORY_PLOT_PATH)

    print("\nEvaluating current config on held-out test split ...")
    best_model = load_model(BEST_MODEL_PATH)
    test_metrics = best_model.evaluate(fit_test_input, y_test_oh, verbose=0)
    y_prob = best_model.predict(fit_test_input, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    cm = confusion_matrix(label_test, y_pred)
    cm_norm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    test_accuracy = float(np.mean(label_test == y_pred))
    test_macro_f1 = float(f1_score(label_test, y_pred, average="macro", zero_division=0))
    test_push_recall = float(
        recall_score(
            label_test,
            y_pred,
            labels=[PUSH_CLASS_ID],
            average=None,
            zero_division=0,
        )[0]
    )

    print("Confusion Matrix:")
    print(cm)
    print("Normalized Confusion Matrix:")
    print(np.around(cm_norm, decimals=2))
    print("Test Accuracy:", test_accuracy)
    print("Test Macro-F1:", test_macro_f1)
    print("Test Push Recall:", test_push_recall)
    print("\nClassification Report:")
    report_text = classification_report(
        label_test,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        label_test,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    print(report_text)

    plot_confusion_matrix_figure(
        cm,
        CLASS_NAMES,
        CONFUSION_PLOT_PATH,
        held_out_accuracy=test_accuracy,
        macro_f1=test_macro_f1,
    )

    prediction_payload = {
        "file_path": file_paths[test_idx].tolist(),
        "true_label": [CLASS_NAMES[idx] for idx in label_test],
        "pred_label": [CLASS_NAMES[idx] for idx in y_pred],
        "correct": (label_test == y_pred).tolist(),
    }
    for class_idx, class_name in enumerate(CLASS_NAMES):
        prediction_payload[f"prob_{class_name}"] = y_prob[:, class_idx].astype(np.float32)
    pd.DataFrame(prediction_payload).to_csv(TEST_PREDICTIONS_PATH, index=False)
    print("Saved test predictions to:", TEST_PREDICTIONS_PATH)

    model_config = {
        "model_type": "keras_dfs_classifier",
        "model_name": TRAIN_MODEL_NAME,
        "class_names": CLASS_NAMES,
        "seed": SEED,
        "data_dir": str(DATA_DIR),
        "output_dir": str(OUTPUT_DIR),
        "input_shape": list(input_shape),
        "t_target": T_TARGET,
        "f_dim": F_DIM,
        "channel_dim": CHANNEL_DIM,
        "normalization": TRAIN_NORMALIZATION,
        "loss": TRAIN_LOSS,
        "use_delta_channel": TRAIN_USE_DELTA_CHANNEL,
        "use_class_weight": TRAIN_USE_CLASS_WEIGHT,
        "use_raw_t_feature": TRAIN_USE_RAW_T_FEATURE,
        "dfs_return_onesided": DFS_RETURN_ONESIDED,
        "dfs_column_normalize": DFS_COLUMN_NORMALIZE,
        "focal_gamma": FOCAL_GAMMA,
        "focal_alpha_mode": FOCAL_ALPHA_MODE,
        "focal_alpha": focal_alpha,
        "train_clap_weight_mult": TRAIN_CLAP_WEIGHT_MULT,
        "train_push_weight_mult": TRAIN_PUSH_WEIGHT_MULT,
        "train_tap_weight_mult": TRAIN_TAP_WEIGHT_MULT,
        "class_weight": class_weight,
        "monitor_metric": MONITOR_METRIC,
        "monitor_mode": MONITOR_MODE,
        "test_size": TEST_SIZE,
        "validation_split_on_trainval": VAL_SPLIT,
        "epochs_requested": N_EPOCHS,
        "epochs_completed": int(len(history_dict.get("loss", []))),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "dropout_ratio": DROPOUT_RATIO,
        "t_target": T_TARGET,
        "filter_length_outliers": FILTER_LENGTH_OUTLIERS,
        "normalization_stats": normalization_stats,
        "raw_t_stats": raw_t_stats,
        "dataset_summary_path": str(DATASET_SUMMARY_PATH),
        "artifacts": {
            "best_model": str(BEST_MODEL_PATH),
            "training_history_plot": str(TRAINING_HISTORY_PLOT_PATH),
            "confusion_matrix_plot": str(CONFUSION_PLOT_PATH),
            "test_predictions_csv": str(TEST_PREDICTIONS_PATH),
            "length_distribution_plot": str(LENGTH_DISTRIBUTION_PLOT_PATH),
            "class_mean_dfs_plot": str(CLASS_MEAN_DFS_PLOT_PATH),
        },
        "history": history_to_float_dict(history),
        "test_metrics": {
            "loss": float(test_metrics[0]),
            "accuracy": test_accuracy,
            "macro_f1": test_macro_f1,
            "push_recall": test_push_recall,
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "classification_report": report_dict,
    }
    MODEL_CONFIG_PATH.write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    print("Saved model config to:", MODEL_CONFIG_PATH)

    print(f"Current config test accuracy = {test_accuracy:.4f}")
    print(f"Current config test macro-F1 = {test_macro_f1:.4f}")
    print(f"Current config push recall = {test_push_recall:.4f}")


if __name__ == "__main__":
    main()
