mport tensorflow as tf
import time

# 1. H100 GPU Hardware Verification
print("\n" + "="*40)
print("NVIDIA H100 SYSTEM CHECK")
print("="*40)
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        details = tf.config.experimental.get_device_details(gpus[0])
        print(f"SUCCESS! Found GPU: {details.get('device_name', 'Unknown')}")
        print(f"Compute Capability: {details.get('compute_capability', 'Unknown')}")
    except Exception as e:
        print(f"Found GPU but could not get details: {e}")
else:
    print("ERROR: No GPU detected. Check container setup.")
    exit()

# 2. Performance Benchmark (Large Matrix Mult)
print("\nStarting Performance Test (Large Matrix Mult)...")
with tf.device('/GPU:0'):
    matrix_size = 15000
    a = tf.random.normal([matrix_size, matrix_size])
    b = tf.random.normal([matrix_size, matrix_size])
    
    # Warm up
    _ = tf.matmul(a, b)
    
    start = time.time()
    c = tf.matmul(a, b)
    _ = c.numpy()
    duration = time.time() - start
    print(f"15k x 15k Matrix Mult took: {duration:.4f} seconds")

# 3. Synthetic Training Load
print("\nStarting Synthetic Training...")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4096, activation='relu', input_shape=(1000,)),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')

x_train = tf.random.normal((50000, 1000))
y_train = tf.random.uniform((50000, 1), maxval=2, dtype=tf.int32)

start_train = time.time()
model.fit(x_train, y_train, epochs=3, batch_size=1024, verbose=1)
print(f"\nTraining Benchmark Complete in: {time.time() - start_train:.2f} seconds")
print("="*40)
