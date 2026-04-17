import json
import numpy as np
import pandas as pd
import h5py
from pathlib import Path

# =========================================================
# 路径
# =========================================================
BASE_DIR = Path(r"F:\Wireless_Sensing\chen_CSI\chen_new_002\chen_new_demo")
CSI_DIR = BASE_DIR / r"wifi_csi\device-ax210-mini02\20260408_075937"
OUT_DIR = BASE_DIR / "demo_segments"
OUT_DIR.mkdir(exist_ok=True)

# =========================================================
# 你在这里填要 trim 的区间
# 直接填 nearest_wifi_idx 范围
# =========================================================
SEGMENTS = [
    {"label": "demo_60s",  "wifi_idx_start": 143, "wifi_idx_end": 2780},
   
    

    # {"label": "push_2",  "wifi_idx_start": 500, "wifi_idx_end": 528},
    # {"label": "clap_2",  "wifi_idx_start": 548, "wifi_idx_end": 560},
]

# 是否同时保存每段包含的原始 h5 文件编号
SAVE_META = False


# =========================================================
# H5 工具
# =========================================================
def list_candidate_datasets(h5_path: Path):
    candidates = []
    with h5py.File(h5_path, "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                candidates.append((name, obj.shape, obj.dtype))
        f.visititems(visitor)
    return candidates


def try_extract_complex_from_dataset(arr):
    arr = np.asarray(arr)

    if np.iscomplexobj(arr):
        return arr

    if arr.ndim >= 1 and arr.shape[-1] == 2:
        real = arr[..., 0]
        imag = arr[..., 1]
        return real + 1j * imag

    if arr.dtype.fields is not None:
        keys = set(arr.dtype.fields.keys())
        if {"real", "imag"}.issubset(keys):
            return arr["real"] + 1j * arr["imag"]
        if {"r", "i"}.issubset(keys):
            return arr["r"] + 1j * arr["i"]

    return None


def extract_csi_from_h5(h5_path: Path):
    candidates = list_candidate_datasets(h5_path)

    priority = []
    fallback = []

    for name, shape, dtype in candidates:
        if "csi" in name.lower() or "channel" in name.lower():
            priority.append((name, shape, dtype))
        else:
            fallback.append((name, shape, dtype))

    ordered = priority + fallback

    with h5py.File(h5_path, "r") as f:
        for name, shape, dtype in ordered:
            arr = f[name][()]
            csi = try_extract_complex_from_dataset(arr)
            if csi is None:
                continue

            csi = np.asarray(csi)
            csi = np.squeeze(csi)

            if csi.ndim == 1:
                return csi

            if csi.ndim == 2:
                if csi.shape[0] >= csi.shape[1]:
                    return csi[:, 0]
                return csi[0, :]

            flat = np.reshape(csi, (-1, csi.shape[-1]))
            if flat.shape[-1] >= flat.shape[0]:
                return flat[0, :]
            return flat[:, 0]

    raise RuntimeError(f"无法从 {h5_path.name} 提取 CSI")


def load_all_csi(csi_dir: Path):
    h5_files = []
    for p in csi_dir.glob("*.h5"):
        if p.stem.isdigit():
            h5_files.append((int(p.stem), p))

    h5_files = sorted(h5_files, key=lambda x: x[0])

    if not h5_files:
        raise FileNotFoundError("没找到 .h5 文件")

    csi_list = []
    file_ids = []

    for file_id, h5_path in h5_files:
        csi_vec = extract_csi_from_h5(h5_path)
        csi_list.append(np.asarray(csi_vec))
        file_ids.append(file_id)

    min_len = min(len(x) for x in csi_list)
    csi_list = [x[:min_len] for x in csi_list]

    csi_mat = np.stack(csi_list, axis=0)   # [T, S]
    return csi_mat, file_ids


# =========================================================
# trim 主流程
# =========================================================
def main():
    csi_mat, file_ids = load_all_csi(CSI_DIR)
    T, S = csi_mat.shape

    print(f"Loaded CSI shape = {csi_mat.shape}")

    summary = []

    for seg in SEGMENTS:
        label = seg["label"]
        s = int(seg["wifi_idx_start"])
        e = int(seg["wifi_idx_end"])

        if s < 0 or e >= T or s > e:
            print(f"[Skip] invalid range: {seg}")
            continue

        csi_seg = csi_mat[s:e+1, :]   # [Tseg, S]

        # 保存 npy
        npy_path = OUT_DIR / f"{label}.npy"
        np.save(npy_path, csi_seg)

        # 保存 mat-like csv metadata
        if SAVE_META:
            meta = {
                "label": label,
                "wifi_idx_start": s,
                "wifi_idx_end": e,
                "num_packets": int(csi_seg.shape[0]),
                "num_subcarriers": int(csi_seg.shape[1]),
                "source_h5_file_start": int(file_ids[s]),
                "source_h5_file_end": int(file_ids[e]),
            }
            with open(OUT_DIR / f"{label}_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

        summary.append({
            "label": label,
            "wifi_idx_start": s,
            "wifi_idx_end": e,
            "num_packets": int(csi_seg.shape[0]),
            "num_subcarriers": int(csi_seg.shape[1]),
            "source_h5_file_start": int(file_ids[s]),
            "source_h5_file_end": int(file_ids[e]),
            "saved_npy": str(npy_path),
        })

        print(f"[OK] {label}: idx [{s}, {e}] -> shape {csi_seg.shape}")

    pd.DataFrame(summary).to_csv(OUT_DIR / "trim_summary.csv", index=False)
    print(f"\nDone. Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()