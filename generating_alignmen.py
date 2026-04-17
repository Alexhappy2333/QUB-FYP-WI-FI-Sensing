import json
import numpy as np
import pandas as pd
from pathlib import Path

# =========================================================
# 路径
# =========================================================
BASE_DIR = Path(r"F:\Wireless_Sensing\chen_CSI\chen_new_002\chen_new_nogesutre")
WIFI_TS_FILE = BASE_DIR / r"wifi_csi\device-ax210-mini02\20260408_080212\merged_timestamps.json"
CAMERA_TS_FILE = BASE_DIR / r"camera\timestamps.csv"
OUT_DIR = BASE_DIR / "analysis_outputs"
OUT_DIR.mkdir(exist_ok=True)

# =========================================================
# 读 WiFi timestamp
# =========================================================
with open(WIFI_TS_FILE, "r", encoding="utf-8") as f:
    wifi_data = json.load(f)

wifi_df = pd.DataFrame(wifi_data)

if "timestamp_sys" not in wifi_df.columns:
    raise ValueError(f"wifi json 没有 timestamp_sys，当前列：{list(wifi_df.columns)}")

if "frame_id" not in wifi_df.columns:
    wifi_df["frame_id"] = np.arange(len(wifi_df))

wifi_df = wifi_df.sort_values("frame_id").reset_index(drop=True)
wifi_df["wifi_idx"] = np.arange(len(wifi_df))

# =========================================================
# 读 Camera timestamp
# =========================================================
cam_df = pd.read_csv(CAMERA_TS_FILE)

if "timestamp_sys" not in cam_df.columns:
    # 如果列名不同，就改这里
    raise ValueError(f"camera csv 没有 timestamp_sys，当前列：{list(cam_df.columns)}")

if "frame_id" not in cam_df.columns:
    cam_df["frame_id"] = np.arange(len(cam_df))

cam_df = cam_df.sort_values("frame_id").reset_index(drop=True)

# =========================================================
# raw timestamp 最近邻对齐
# =========================================================
wifi_ts = wifi_df["timestamp_sys"].values.astype(np.int64)
cam_ts = cam_df["timestamp_sys"].values.astype(np.int64)

idx = np.searchsorted(wifi_ts, cam_ts)

idx_left = np.clip(idx - 1, 0, len(wifi_ts) - 1)
idx_right = np.clip(idx, 0, len(wifi_ts) - 1)

err_left = np.abs(cam_ts - wifi_ts[idx_left])
err_right = np.abs(cam_ts - wifi_ts[idx_right])

choose_right = err_right < err_left
nearest_wifi_idx = np.where(choose_right, idx_right, idx_left)
nearest_err_raw = np.abs(cam_ts - wifi_ts[nearest_wifi_idx])

cam_df["nearest_wifi_idx"] = nearest_wifi_idx
cam_df["nearest_wifi_timestamp_sys"] = wifi_ts[nearest_wifi_idx]
cam_df["align_err_raw"] = nearest_err_raw

# =========================================================
# 保存结果
# =========================================================
out_file = OUT_DIR / "camera_to_wifi_raw_alignment.csv"
cam_df.to_csv(out_file, index=False)

print("Done.")
print(f"saved: {out_file}")
print()
print("alignment raw error stats:")
print(f"mean   = {cam_df['align_err_raw'].mean():.2f}")
print(f"median = {cam_df['align_err_raw'].median():.2f}")
print(f"95%    = {cam_df['align_err_raw'].quantile(0.95):.2f}")
print(f"max    = {cam_df['align_err_raw'].max():.2f}")