# MobileNet Bird Classification Example

## 1. Software Abstract

This example uses **MobileNet**, a lightweight, pre-trained deep learning model, to classify bird images. MobileNet is widely used in mobile and edge computing due to its small size and speed. It’s useful in scientific and engineering applications like wildlife monitoring, field data collection, and real-time classification on embedded devices.

This is a **pre-trained programming tool** used for image classification. The goal of this project is to explore how a pre-trained MobileNet model can be used to identify birds from trail camera images for a wildlife monitoring project.

---

## 2. Installation

### Local Installation (Recommended for Testing)

```bash
pip install tensorflow opencv-python pillow numpy

This installs:

- `tensorflow` – loads and runs MobileNet  
- `opencv-python` – (optional) for image preprocessing  
- `pillow` – lightweight image handling  
- `numpy` – array manipulation  
```

### HPCC Installation (Using Virtual Environment)

```bash
module load python
python -m venv myenv
source myenv/bin/activate
pip install tensorflow opencv-python pillow numpy
```
To check if it worked:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```
---

## 4. Submission Script

To run this example on the HPCC, save the following script as `run_classifier.sb`:

```bash
#!/bin/bash
#SBATCH --job-name=bird_classifier
#SBATCH --output=output.txt
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --mem=2G

module load python
source myenv/bin/activate
python bird_classifier.py

---

## 5. References

- [TensorFlow MobileNetV2 Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2)
- [Keras Applications Overview](https://keras.io/api/applications/)
- [ImageNet Class Index Labels](https://www.image-net.org/)
- [Pillow (Python Imaging Library)](https://pillow.readthedocs.io/en/stable/)
- [Bird Image – Green-Cheeked Conure]([https://commons.wikimedia.org/wiki/File:Green-cheeked_Conure_-_Florida_-_USA_S4E5161_(23303264352).jpg](https://www.petguide.com/breeds/bird/green-cheeked-conure/))
