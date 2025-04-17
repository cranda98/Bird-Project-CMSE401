import pandas as pd
import numpy as np
import os
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import tensorflow as tf

model = MobileNetV2(weights='imagenet')

df_gt = pd.read_csv('ground_truth.csv')

serial_preds = []
batch_preds = []
true_labels = []
filenames = []
images = []

for _, row in df_gt.iterrows():
    fname = row['filename']
    true_label = row['correct_label'].lower()
    img_path = os.path.join('./Birds', fname)

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_preprocessed = preprocess_input(np.expand_dims(img_array, axis=0))

    filenames.append(fname)
    true_labels.append(true_label)
    images.append(img_preprocessed)

for img in images:
    preds = model.predict(img, verbose=0)
    label = decode_predictions(preds, top=1)[0][0][1].lower()
    serial_preds.append(label)

batch_input = np.vstack(images)
batch_output = model.predict(batch_input, verbose=0)
for pred in batch_output:
    label = decode_predictions(np.expand_dims(pred, axis=0), top=1)[0][0][1].lower()
    batch_preds.append(label)

results = pd.DataFrame({
    'img_id': filenames,
    'true_label': true_labels,
    'serial_pred': serial_preds,
    'batch_pred': batch_preds
})
results['serial_correct'] = results['true_label'] == results['serial_pred']
results['batch_correct'] = results['true_label'] == results['batch_pred']

serial_acc = results['serial_correct'].mean()
batch_acc = results['batch_correct'].mean()

print("=== Human-Labeled Accuracy Comparison ===")
print(results[['img_id', 'true_label', 'serial_pred', 'batch_pred', 'serial_correct', 'batch_correct']])
print(f"\nSerial Top-1 Accuracy: {serial_acc:.2%}")
print(f"Batch Top-1 Accuracy:  {batch_acc:.2%}")
