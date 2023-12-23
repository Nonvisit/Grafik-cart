# Grafik-cart
Grafik cart
# Code for Multiple GPU Connections
import tensorflow as tf

# Check available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    # If multiple GPUs are available, connect to all of them
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(f"Connected to {len(logical_gpus)} GPUs: {logical_gpus}")
else:
    print("No GPUs available.")
