import os
import gzip
import requests
from tqdm import tqdm

# ---------------------- 配置 ----------------------
URL = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Sports_and_Outdoors.jsonl.gz"
OUT_GZ = "meta_sports.jsonl.gz"
OUT_JSONL = "meta_sports.jsonl"
# ---------------------------------------------------

def download_file(url, out_path):
    if os.path.exists(out_path):
        print(f"✅ {out_path} 已存在，跳过下载")
        return
    print(f"📥 开始下载：{url}")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with tqdm(total=total, unit="B", unit_scale=True, desc=out_path) as pbar:
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    print(f"✅ 下载完成：{out_path}")

def extract_gz(gz_path, jsonl_path):
    if os.path.exists(jsonl_path):
        print(f"✅ {jsonl_path} 已存在，跳过解压")
        return
    print(f"🔧 解压 {gz_path} → {jsonl_path}")
    with gzip.open(gz_path, "rb") as f_in, open(jsonl_path, "wb") as f_out:
        data = f_in.read()
        f_out.write(data)
    print(f"✅ 解压完成：{jsonl_path}")

if __name__ == "__main__":
    download_file(URL, OUT_GZ)
    extract_gz(OUT_GZ, OUT_JSONL)
    print("\n🎉 全部完成！现在可以解析 meta_sports.jsonl 拿到官方类目")