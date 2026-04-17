import json
import glob
import os
import pandas as pd

# ===================== 【关键】强制指定你的文件夹路径 =====================
# 这一行必须改成你所有 json 文件所在的目录！
FOLDER_PATH = r"F:\Wireless_Sensing\chen_CSI\chen_new_002\chen_new_nogesutre\wifi_csi\device-ax210-mini02\20260408_080212"

# 匹配该目录下所有 xxx_sync.json
json_pattern = os.path.join(FOLDER_PATH, "*_sync.json")
json_files = glob.glob(json_pattern)

# 按数字排序（1,2,3...2449）
json_files.sort(key=lambda x: int(os.path.basename(x).split("_")[0]))

print(f"找到文件总数：{len(json_files)}")

data_list = []

# 遍历合并
for idx, file_path in enumerate(json_files, start=0):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        timestamp_sys = data.get("timestamp_sys")
        
        data_list.append({
            "frame_id": idx,
            "timestamp_sys": timestamp_sys
        })
    except Exception as e:
        print(f"失败：{file_path} => {str(e)}")

# 输出
df = pd.DataFrame(data_list)

# 输出 JSON
df.to_json(
    os.path.join(FOLDER_PATH, "merged_timestamps.json"),
    orient="records",
    indent=2
)

 

print("="*50)
print(f"✅ 合并成功！总数：{len(df)} 条")
print(f"📄 已保存到：{FOLDER_PATH}")
print(f"📋 列：frame_id, timestamp_sys")