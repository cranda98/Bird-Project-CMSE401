import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import time

model = MobileNetV2(weights='imagenet')
print("MobileNetV2 model loaded.")

folder_path = './Birds'
inference_times = []
image_count = 0

total_start = time.time()

for img_file in sorted(os.listdir(folder_path)):
    if img_file.lower().endswith(('.jpeg', '.jpg', '.png')):
        img_path = os.path.join(folder_path, img_file)
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            start = time.time()
            preds = model.predict(img_array, verbose=0)
            end = time.time()

            inference_times.append(end - start)
            image_count += 1

            decoded = decode_predictions(preds, top=3)[0]
            print(f"\n{img_file} predictions:")
            for i, (imagenetID, label, prob) in enumerate(decoded):
                print(f"  {i + 1}. {label} ({prob:.4f})")
        except Exception as e:
            print(f"Error processing {img_file}: {e}")

total_end = time.time()
total_time = total_end - total_start

if image_count > 0:
    avg_time = np.mean(inference_times)
    print(f"\nProcessed {image_count} images.")
    print(f"Total inference time: {total_time:.4f} seconds")
    print(f"Average inference time per image: {avg_time:.4f} seconds")
else:
    print("No valid images found.")
