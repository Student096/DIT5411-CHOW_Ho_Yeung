import tensorflow as tf

print("TensorFlow 版本:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("偵測到 GPU:", gpus)

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    with tf.device("/GPU:0"):
        a = tf.ones((256, 256))
        b = tf.ones((256, 256))
        c = tf.matmul(a, b)
    print("GPU 矩陣運算測試成功，輸出形狀:", c.shape)
else:
    print("未偵測到 GPU，目前將以 CPU 執行。")
