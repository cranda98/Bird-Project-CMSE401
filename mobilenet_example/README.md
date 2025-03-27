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

To check if it worked:

python -c "import tensorflow as tf; print(tf.__version__)"
```
