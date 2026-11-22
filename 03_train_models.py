import os
import time
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

# ================== 參數設定 ==================
WORK_DIR = os.path.expanduser("~/DIT5411-your-name")
DATA_ROOT = os.path.join(WORK_DIR, "dataset_extracted_all")
CKPT_DIR = os.path.join(WORK_DIR, "checkpoints")

IMG_SIZE = 64
NUM_CLASSES = 13065       # 功課規定輸出層 13,065 個神經元
TRAIN_PER_CHAR = 40       # 功課規定頭 40 張做 training set
MIN_TRAIN_PER_CHAR = 200  # 增強後每個字 training set 最少 200 張
BATCH_SIZE = 128
EPOCHS = 9                # 訓練回合數設定為 9
MAX_CLASSES = None        # None = 跑全部 13,065 類；快速測試可改為 100
SEED = 42
# ================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)


def setup_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("偵測到 GPU:", gpus)


def scan_character_dirs(root_dir: str):
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    dir_map = {}

    for curr_root, _, files in os.walk(root_dir):
        imgs = sorted([
            os.path.join(curr_root, f)
            for f in files
            if os.path.splitext(f)[1].lower() in valid_exts
        ])
        if imgs:
            cname = os.path.basename(curr_root)
            if cname not in dir_map:
                dir_map[cname] = imgs
            else:
                dir_map[cname].extend(imgs)

    sorted_cnames = sorted(dir_map.keys())
    if MAX_CLASSES is not None:
        sorted_cnames = sorted_cnames[:MAX_CLASSES]

    return [(cname, sorted(dir_map[cname])) for cname in sorted_cnames]


def load_and_resize_gray(img_path: str, size: int = IMG_SIZE) -> np.ndarray:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.ones((size, size), dtype=np.uint8) * 255
    if img.shape[0] != size or img.shape[1] != size:
        interp = cv2.INTER_AREA if (img.shape[0] >= size and img.shape[1] >= size) else cv2.INTER_LINEAR
        img = cv2.resize(img, (size, size), interpolation=interp)
    return img


def augment_char_image(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """對單張灰階手寫字做 rotation（旋轉）、scaling（縮放）、shearing（錯切）"""
    h, w = img.shape[:2]
    angle = float(rng.uniform(-15.0, 15.0))
    scale = float(rng.uniform(0.85, 1.15))
    shear_x = float(rng.uniform(-0.15, 0.15))
    shear_y = float(rng.uniform(-0.15, 0.15))

    center = (w / 2.0, h / 2.0)
    M_rot_scale = cv2.getRotationMatrix2D(center, angle, scale)
    M_shear = np.array([
        [1.0, shear_x, -shear_x * center[0]],
        [shear_y, 1.0, -shear_y * center[1]]
    ], dtype=np.float32)

    M1 = np.vstack([M_rot_scale, [0.0, 0.0, 1.0]])
    M2 = np.vstack([M_shear, [0.0, 0.0, 1.0]])
    M_comb = (M2 @ M1)[:2, :]

    bg_val = int(np.median(img))
    aug = cv2.warpAffine(
        img,
        M_comb,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bg_val
    )
    return aug


def build_records(char_items):
    train_records = []  # (img_path, label_idx, is_aug, aug_seed)
    test_records = []   # (img_path, label_idx)

    for idx, (_, files) in enumerate(char_items):
        if not files:
            continue

        t_files = files[:TRAIN_PER_CHAR]
        e_files = files[TRAIN_PER_CHAR:]

        for p in t_files:
            train_records.append((p, idx, False, 0))

        needed_aug = max(0, MIN_TRAIN_PER_CHAR - len(t_files))
        for k in range(needed_aug):
            src_p = t_files[k % len(t_files)]
            train_records.append((src_p, idx, True, idx * 100000 + k + 1))

        for p in e_files:
            test_records.append((p, idx))

    return train_records, test_records


class CharDataSequence(tf.keras.utils.Sequence):
    def __init__(self, records, batch_size=BATCH_SIZE, shuffle=True, is_train=True, **kwargs):
        super().__init__(**kwargs)
        self.records = records
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.is_train = is_train
        self.indexes = np.arange(len(self.records))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.records) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_idx = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        X = np.empty((len(batch_idx), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
        y = np.empty((len(batch_idx),), dtype=np.int32)

        for i, rec_i in enumerate(batch_idx):
            if self.is_train:
                path, label, is_aug, seed = self.records[rec_i]
                img = load_and_resize_gray(path, IMG_SIZE)
                if is_aug:
                    rng = np.random.default_rng(seed)
                    img = augment_char_image(img, rng)
            else:
                path, label = self.records[rec_i]
                img = load_and_resize_gray(path, IMG_SIZE)

            X[i, :, :, 0] = img.astype(np.float32) / 255.0
            y[i] = label

        return X, y


def build_mlp_model():
    return models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.Flatten(),
        layers.Dense(1024, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(512, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ], name="Model_1_MLP")


def build_basic_cnn_model():
    return models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(512, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ], name="Model_2_Basic_CNN")


def build_deep_cnn_model():
    return models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ], name="Model_3_Deep_CNN_BN")


def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    setup_gpu()

    char_items = scan_character_dirs(DATA_ROOT)
    print(f"載入類別數: {len(char_items)}（模型輸出層固定為 {NUM_CLASSES} 個神經元）")

    train_records, test_records = build_records(char_items)
    print(f"Training 總樣本數: {len(train_records)}（每個字最少 {MIN_TRAIN_PER_CHAR} 張）")
    print(f"Testing 總樣本數: {len(test_records)}")

    train_seq = CharDataSequence(train_records, batch_size=BATCH_SIZE, shuffle=True, is_train=True)
    test_seq = CharDataSequence(test_records, batch_size=BATCH_SIZE, shuffle=False, is_train=False)

    model_builders = [
        ("Model_1_MLP", build_mlp_model),
        ("Model_2_Basic_CNN", build_basic_cnn_model),
        ("Model_3_Deep_CNN_BN", build_deep_cnn_model),
    ]

    results = []

    for m_name, builder in model_builders:
        print(f"\n==================== 開始訓練: {m_name} (EPOCHS={EPOCHS}) ====================")
        t0 = time.time()
        tf.keras.backend.clear_session()
        model = builder()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        model.summary()

        ckpt_path = os.path.join(CKPT_DIR, f"{m_name}_best.weights.h5")
        ckpt_cb = callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )

        history = model.fit(
            train_seq,
            epochs=EPOCHS,
            callbacks=[ckpt_cb],
            verbose=1
        )

        if os.path.exists(ckpt_path):
            model.load_weights(ckpt_path)

        test_loss, test_acc = model.evaluate(test_seq, verbose=1)
        best_train_acc = max(history.history["accuracy"])
        final_train_acc = history.history["accuracy"][-1]
        elapsed = time.time() - t0
        print(
            f"[{m_name}] Best Train Acc: {best_train_acc:.4f} | "
            f"Final Train Acc: {final_train_acc:.4f} | Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.4f} | Time: {elapsed:.1f}s"
        )

        results.append({
            "Model": m_name,
            "Output_Neurons": NUM_CLASSES,
            "Epochs": EPOCHS,
            "Train_Samples": len(train_records),
            "Test_Samples": len(test_records),
            "Best_Train_Accuracy": round(float(best_train_acc), 4),
            "Final_Train_Accuracy": round(float(final_train_acc), 4),
            "Test_Loss": round(float(test_loss), 4),
            "Test_Accuracy": round(float(test_acc), 4),
            "Train_Seconds": round(elapsed, 1),
        })

    df_results = pd.DataFrame(results).sort_values(by="Test_Accuracy", ascending=False)
    csv_path = os.path.join(WORK_DIR, "model_results.csv")
    df_results.to_csv(csv_path, index=False)
    print("\n=== 模型測試結果排名 ===")
    print(df_results.to_string(index=False))
    print("最佳模型:", df_results.iloc[0]["Model"])
    print("結果已儲存至:", csv_path)


if __name__ == "__main__":
    main()
