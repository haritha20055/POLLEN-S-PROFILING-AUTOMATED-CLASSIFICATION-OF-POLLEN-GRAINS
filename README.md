# 🌿 Pollen's Profiling
### Automated Classification of Pollen Grains using Deep Learning

![screenshot](static/last_uploaded.jpg)

## 🚀 Overview
This project is part of my summer internship:  
> **"Pollen's Profiling: Automated Classification of Pollen Grains"**

It uses a Convolutional Neural Network (CNN) to classify microscopic images of pollen grains into 23 classes, and provides a user-friendly **Flask web app** for predictions.

---

## ✅ Features
- 📊 **Data preprocessing & EDA** on pollen dataset
- 🧠 **CNN model** with Keras & TensorFlow, achieving ~75% test accuracy
- 📈 Accuracy & loss plots during training
- 🌐 **Flask web app**: upload an image, see predicted pollen type
- 📝 **PDF report generator**: download prediction summary

---

## 🗂️ Dataset
- ~790 images of pollen grains
- 23 classes (e.g. *arecaceae, anadenanthera, syagrus, urochloa* etc.)
- Images resized to **128x128**, normalized.

---

## ⚙️ Tech Stack
- Python, Numpy, Pandas
- OpenCV, Matplotlib
- TensorFlow + Keras
- Flask (for web app)
- ReportLab (for PDF generation)

---

## 🚀 How to run locally
✅ Clone the repo:
```bash
git clone https://github.com/haritha20055/POLLEN-S-PROFILING-AUTOMATED-CLASSIFICATION-OF-POLLEN-GRAINS.git
cd POLLEN-S-PROFILING-AUTOMATED-CLASSIFICATION-OF-POLLEN-GRAINS
