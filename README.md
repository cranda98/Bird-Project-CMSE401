# AI-Based Bird Species Identification 🐦📸

This project was developed for **CMSE401 (Data Science Capstone)**.  
It uses **deep learning** and **OpenCV** to automatically identify bird species from trail camera images, making bird monitoring faster and easier for conservation efforts.  

---

## 📄 Project Proposal  
Full proposal available here:  
- [📑 `project_proposal.pdf`](project_proposal/project_proposal.pdf)  

---

## 🛠️ Technologies  
- **Python** – Core programming language  
- **TensorFlow** – Deep learning framework  
- **OpenCV** – Image preprocessing and computer vision  
- **MobileNet / ResNet** – Pre-trained CNN models for classification  

---

## 📊 Results & Highlights  
- Successfully classified multiple bird species from image datasets.  
- Compared performance of **CPU vs. GPU inference**, showing major speedups with parallel processing.  
- Implemented **batch image classification** for efficient large-scale analysis.  
- Delivered clear visualizations and metrics demonstrating model accuracy.  

---

## 🚀 How It Works  
1. **Preprocessing** – Images are cleaned and resized with OpenCV.  
2. **Model Inference** – Pre-trained CNN (MobileNet/ResNet) classifies the species.  
3. **Batch Pipeline** – Handles multiple images at once for large datasets.  
4. **Performance Benchmarking** – CPU and GPU run times compared.  

Example command:  
```bash
python classify.py --input sample_images/ --model mobilenet
```
---

## Repository Structure

├── project_proposal/         # Proposal and documentation

├── data/                     # Datasets (not included in repo)

├── notebooks/                # Jupyter notebooks for exploration & training

├── src/                      # Core scripts for preprocessing & classification

├── results/                  # Plots, benchmarks, and outputs

└── README.md                 # Project overview

---

✅ Project Status: Complete
📌 Developed as part of CMSE401, Spring 2025
