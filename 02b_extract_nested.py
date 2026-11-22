import os
import shutil
import subprocess
import zipfile

SRC_DIR = os.path.expanduser("~/DIT5411-your-name/dataset_raw/handwritting_data_all-master")
OUT_DIR = os.path.expanduser("~/DIT5411-your-name/dataset_extracted")


def decode_zip_name(info: zipfile.ZipInfo) -> str:
    if info.flag_bits & 0x800:
        return info.filename
    raw = info.filename.encode("cp437", errors="replace")
    for enc in ("cp950", "big5", "utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return info.filename


def extract_all_archives(src_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(os.listdir(src_dir))
    print(f"找到 {len(files)} 個檔案: {files}")

    for fname in files:
        fpath = os.path.join(src_dir, fname)
        if not os.path.isfile(fpath):
            continue
        low = fname.lower()
        target_sub = os.path.join(out_dir, os.path.splitext(fname)[0])
        os.makedirs(target_sub, exist_ok=True)

        if low.endswith(".zip"):
            print(f"解壓 ZIP: {fname} -> {target_sub}")
            with zipfile.ZipFile(fpath, "r") as zf:
                for info in zf.infolist():
                    rel = decode_zip_name(info).lstrip("/\\")
                    dst = os.path.abspath(os.path.join(target_sub, rel))
                    if not dst.startswith(os.path.abspath(target_sub) + os.sep):
                        continue
                    if info.is_dir():
                        os.makedirs(dst, exist_ok=True)
                    else:
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        with zf.open(info, "r") as s, open(dst, "wb") as d:
                            d.write(s.read())
        elif low.endswith(".rar"):
            if shutil.which("unrar"):
                print(f"解壓 RAR (unrar): {fname} -> {target_sub}")
                subprocess.run(["unrar", "x", "-o+", fpath, target_sub + "/"], check=False)
            elif shutil.which("7z"):
                print(f"解壓 RAR (7z): {fname} -> {target_sub}")
                subprocess.run(["7z", "x", fpath, f"-o{target_sub}", "-y"], check=False)
            else:
                print(f"提示: {fname} 是 RAR 檔，但系統未裝 unrar 或 7z。")
        elif low.endswith(".7z"):
            if shutil.which("7z"):
                print(f"解壓 7Z: {fname} -> {target_sub}")
                subprocess.run(["7z", "x", fpath, f"-o{target_sub}", "-y"], check=False)
            else:
                print(f"提示: {fname} 是 7Z 檔，但系統未裝 7z。")
        else:
            print(f"複製其他檔案: {fname}")
            shutil.copy2(fpath, os.path.join(target_sub, fname))


def inspect_images(root_dir: str) -> None:
    img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    char_counts = {}
    for root, dirs, files in os.walk(root_dir):
        imgs = [f for f in files if os.path.splitext(f)[1].lower() in img_exts]
        if imgs:
            label = os.path.basename(root)
            char_counts[label] = char_counts.get(label, 0) + len(imgs)

    print(f"\n偵測到包含圖片嘅類別資料夾數量: {len(char_counts)}")
    if char_counts:
        sorted_items = sorted(char_counts.items(), key=lambda x: x[1])
        print("樣本數最少嘅前 5 個類別:", sorted_items[:5])
        print("樣本數最多嘅後 5 個類別:", sorted_items[-5:])
        print("前 10 個類別名稱與圖片數:", list(char_counts.items())[:10])


if __name__ == "__main__":
    extract_all_archives(SRC_DIR, OUT_DIR)
    inspect_images(OUT_DIR)
