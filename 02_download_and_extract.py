import os
import zipfile
import urllib.request

REPO_ZIP_URL = "https://github.com/chenkenanalytic/handwritting_data_all/archive/refs/heads/master.zip"
WORK_DIR = os.path.expanduser("~/DIT5411-your-name")
ZIP_PATH = os.path.join(WORK_DIR, "handwritting_data_all.zip")
EXTRACT_DIR = os.path.join(WORK_DIR, "dataset_raw")


def download_file(url: str, dest: str) -> None:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"已存在壓縮檔，略過下載: {dest}")
        return
    print(f"正在下載資料集:\n{url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
        total = resp.headers.get("Content-Length")
        downloaded = 0
        chunk_size = 1024 * 1024
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 / int(total)
                print(f"\r下載進度: {downloaded / (1024*1024):.1f} MB / {int(total)/(1024*1024):.1f} MB ({pct:.1f}%)", end="")
            else:
                print(f"\r已下載: {downloaded / (1024*1024):.1f} MB", end="")
    print("\n下載完成。")


def decode_zip_member_name(info: zipfile.ZipInfo) -> str:
    # ZIP 若未設 UTF-8 flag (bit 11)，傳統中文 zip 多為 cp437 儲存的 cp950/big5 位元組
    if info.flag_bits & 0x800:
        return info.filename
    raw_bytes = info.filename.encode("cp437", errors="replace")
    for enc in ("cp950", "big5", "utf-8", "gbk"):
        try:
            return raw_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    return info.filename


def extract_zip(zip_path: str, extract_dir: str) -> None:
    os.makedirs(extract_dir, exist_ok=True)
    print(f"正在解壓至: {extract_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        total = len(members)
        for idx, info in enumerate(members, start=1):
            rel_name = decode_zip_member_name(info)
            rel_name = rel_name.lstrip("/\\")
            target_path = os.path.abspath(os.path.join(extract_dir, rel_name))
            base_path = os.path.abspath(extract_dir)
            if not target_path.startswith(base_path + os.sep) and target_path != base_path:
                continue
            if info.is_dir():
                os.makedirs(target_path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with zf.open(info, "r") as src, open(target_path, "wb") as dst:
                    dst.write(src.read())
            if idx % 5000 == 0 or idx == total:
                print(f"\r解壓進度: {idx} / {total}", end="")
    print("\n解壓完成。")


def inspect_dataset(extract_dir: str) -> None:
    img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    char_counts = {}

    for root, dirs, files in os.walk(extract_dir):
        imgs = [f for f in files if os.path.splitext(f)[1].lower() in img_exts]
        if imgs:
            label = os.path.basename(root)
            char_counts[label] = char_counts.get(label, 0) + len(imgs)

    print(f"偵測到包含圖片嘅類別資料夾數量: {len(char_counts)}")
    if char_counts:
        sorted_items = sorted(char_counts.items(), key=lambda x: x[1])
        print("樣本數最少嘅前 5 個類別:", sorted_items[:5])
        print("樣本數最多嘅後 5 個類別:", sorted_items[-5:])
        sample_preview = list(char_counts.items())[:10]
        print("前 10 個類別名稱與圖片數:", sample_preview)
    else:
        print("暫時未喺子資料夾直接搵到圖片，請檢查解壓目錄結構：")
        for root, dirs, files in os.walk(extract_dir):
            print(" ", root, "dirs:", len(dirs), "files:", len(files))
            if root.count(os.sep) - extract_dir.count(os.sep) >= 2:
                break


if __name__ == "__main__":
    download_file(REPO_ZIP_URL, ZIP_PATH)
    extract_zip(ZIP_PATH, EXTRACT_DIR)
    inspect_dataset(EXTRACT_DIR)
