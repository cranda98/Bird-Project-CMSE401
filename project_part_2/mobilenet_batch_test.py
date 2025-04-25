import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import time

print("GPUs available:", tf.config.list_physical_devices('GPU'))

model = MobileNetV2(weights='imagenet')

folder_path = './Birds'
batch_images = []
filenames = []

for img_file in os.listdir(folder_path):
    if img_file.lower().endswith(('.jpeg', '.jpg', '.png')):
        img_path = os.path.join(folder_path, img_file)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        batch_images.append(img_array)
        filenames.append(img_file)

batch_array = np.array(batch_images)

start = time.time()
preds = model.predict(batch_array)
end = time.time()

for i, pred in enumerate(preds):
    decoded = decode_predictions(np.expand_dims(pred, axis=0), top=3)[0]
    print(f"\n{filenames[i]} predictions:")
    for j, (imagenetID, label, prob) in enumerate(decoded):
        print(f"  {j + 1}. {label} ({prob:.4f})")

print(f"\nTotal batch time: {end - start:.4f} seconds")
print(f"Average time per image (batch): {(end - start)/len(batch_images):.4f} seconds")
