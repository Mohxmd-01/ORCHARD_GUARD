import tensorflow as tf

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))    
for device in tf.config.list_physical_devices():
    print(device)
