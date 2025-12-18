import os
import zipfile

import requests
from tqdm import tqdm

# --- 設定 ---
DATA_ROOT = "./data/nist"
EXTRACT_ROOT = os.path.join(DATA_ROOT, "extracted")

# 修正点: 辞書(URLS)をやめ、単一の定数に変更しました
BY_CLASS_URL = "https://s3.amazonaws.com/nist-srd/SD19/by_class.zip"


def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"[Skip] {os.path.basename(dest_path)} already exists.")
        return

    print(f"[Download] Downloading {url} ...")
    try:
        r = requests.get(url, stream=True)
        total = int(r.headers.get("content-length", 0))
        with (
            open(dest_path, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, ncols=80) as bar,
        ):
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        print("Done.")
    except Exception as e:
        print(f"[Error] Failed to download {url}: {e}")


def extract_file(zip_path, extract_to):
    print(f"[Extract] Extracting {os.path.basename(zip_path)} into {extract_to} ...")
    os.makedirs(extract_to, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            for member in tqdm(z.infolist(), desc="Files", ncols=80):
                z.extract(member, extract_to)
        print("Done.")
    except zipfile.BadZipFile:
        print(f"[Error] Bad zip file: {zip_path}")


def main():
    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(EXTRACT_ROOT, exist_ok=True)

    # by_class を extracted に解凍
    zip_class = os.path.join(DATA_ROOT, "by_class.zip")

    # 修正点: 定数を直接渡すように変更
    download_file(BY_CLASS_URL, zip_class)
    extract_file(zip_class, EXTRACT_ROOT)

    print("\n" + "=" * 50)
    print("Setup Complete!")
    print(f"Files extracted into: {EXTRACT_ROOT}")
    print("Structure should be:")
    print(f"  {EXTRACT_ROOT}/by_class")
    print("=" * 50)


if __name__ == "__main__":
    main()
