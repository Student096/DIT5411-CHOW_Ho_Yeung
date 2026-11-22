import glob
import os
import zipfile

SRC_DIR = os.path.expanduser("~/DIT5411-your-name/dataset_raw/handwritting_data_all-master")
MERGED_ZIP = os.path.expanduser("~/DIT5411-your-name/dataset_merged/all_data.zip")
OUT_DIR = os.path.expanduser("~/DIT5411-your-name/dataset_extracted_all")


def merge_split_zip(src_dir: str, merged_zip: str) -> None:
    parts = sorted(glob.glob(os.path.join(src_dir, "all_data.zip.*")))
    if not parts:
        raise FileNotFoundError(f"搵唔到 all_data.zip.* 分割檔於: {src_dir}")

    os.makedirs(os.path.dirname(merged_zip), exist_ok=True)
    print(f"找到 {len(parts)} 個分割檔，開始合併至: {merged_zip}")

    total_bytes = 0
    with open(merged_zip, "wb") as out:
        for p in parts:
            sz = os.path.getsize(p)
            total_bytes += sz
            print(f"  合併: {os.path.basename(p)} ({sz / (1024*1024):.1f} MB)")
            with open(p, "rb") as src:
                while True:
                    chunk = src.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
    print(f"合併完成，總大小: {total_bytes / (1024*1024):.1f} MB")


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


def extract_merged_zip(merged_zip: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    print(f"開始解壓 {merged_zip} 至: {out_dir}")
    with zipfile.ZipFile(merged_zip, "r") as zf:
        members = zf.infolist()
        total = len(members)
        for idx, info in enumerate(members, start=1):
            rel = decode_zip_name(info).lstrip("/\\")
            dst = os.path.abspath(os.path.join(out_dir, rel))
            if not dst.startswith(os.path.abspath(out_dir) + os.sep):
                continue
            if info.is_dir():
                os.makedirs(dst, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with zf.open(info, "r") as s, open(dst, "wb") as d:
                    d.write(s.read())
            if idx % 10000 == 0 or idx == total:
                print(f"\r解壓進度: {idx} / {total}", end="")
    print("\n解壓完成。")


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
    merge_split_zip(SRC_DIR, MERGED_ZIP)
    extract_merged_zip(MERGED_ZIP, OUT_DIR)
    inspect_images(OUT_DIR)
