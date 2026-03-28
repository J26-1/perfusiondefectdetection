# Perfusion AI: Interpretable Deep Learning Framework for Myocardial Perfusion SPECT Defect Detection
🔗 **Live Demo:** https://perfusiondefectdetection.streamlit.app/  
📂 **Repository:** https://github.com/J26-1/perfusiondefectdetection  

## Overview
PerfusionAI is an interpretable deep learning–based clinical decision support system for analyzing myocardial perfusion SPECT images. It performs automatic myocardial segmentation and assists in detecting perfusion defects while providing transparent and uncertainty-aware predictions.

Designed to address the limitations of manual interpretation—such as noise, attenuation artifacts, and inter-observer variability—PerfusionAI integrates explainability and region-based analysis to support reliable clinical decision-making.

## Features
- **U-Net Segmentation** for precise myocardial localization  
- **Perfusion Defect Detection** within segmented myocardium  
- **Grad-CAM Explainability** to visualize model attention  
- **Uncertainty Estimation** using Monte Carlo Dropout  
- **Polar Map Visualization** for region-wise perfusion analysis  
- **Region-Based Evaluation** (anterior, mid, inferior)  


## Output
Try the live application:  
https://perfusiondefectdetection.streamlit.app/

Example outputs include:
- Segmentation maps  
- Overlay (prediction vs ground truth)  
- Error maps (false positives / false negatives)  
- Grad-CAM visualizations  
- Confidence and uncertainty maps  
- Polar maps for regional perfusion analysis  

*(Add screenshots here for best impact)*

---

## Dataset
This project uses the **Myocardial Perfusion SPECT dataset** from PhysioNet.

- **Format:** DICOM images + NIfTI masks  
- **Task:** Myocardial segmentation and perfusion analysis  
- **Constraint:** No external labeled datasets used  

### Citation
Calixto, W., Nogueira, S., Luz, F., & Ortiz de Camargo, T. F. (2025).  
*Myocardial perfusion scintigraphy image database (version 1.0.0).* PhysioNet.  
https://doi.org/10.13026/ce2z-dw74  

## Installation
```bash
git clone https://github.com/J26-1/perfusiondefectdetection.git
cd perfusiondefectdetection
pip install -r requirements.txt
```

## How To Run
```bash
python main.py    # Run Model
streamlit run app.py    # Run Streamlit App
```

## Tech Stack
- **Programming Language:** Python  
- **Deep Learning Framework:** PyTorch  
- **Medical Image Processing:** NumPy, OpenCV, PIL  
- **Visualization:** Matplotlib, Plotly  
- **Explainability:** Grad-CAM  
- **Uncertainty Estimation:** Monte Carlo Dropout  
- **Web Framework:** Streamlit  
- **Data Format:** DICOM, NIfTI  
- **Model:** U-net architecture
- **Loss:** BCE + Dice  
- **Optimizer:** Adam
- **Training:** Stratified split, LR scheduling, Early stopping
