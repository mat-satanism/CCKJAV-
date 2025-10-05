# CCKJAV-
---

# 🚀 **AI-Driven Exoplanet Discovery System**

## 🌍 **Team Vision**

We are a multidisciplinary team combining **data science, astrophysics, and visualization** to accelerate the search for planets beyond our Solar System.
Our goal is to build an end-to-end **AI system** that automatically identifies **exoplanet candidates** using open-source data from NASA missions — **Kepler (KOI)**, **TESS (TOI)**, and others — unifying them into one transparent and explainable machine-learning pipeline.

---

## 🧭 **What Our System Does**

Our platform analyzes real astrophysical data and classifies observed signals as either:

* **Confirmed Planet**,
* **Candidate**, or
* **False Positive**.

The system combines **data cleaning, feature engineering, supervised ML modeling, and interactive visualization** to provide both researchers and the public with a fast and interpretable way to explore exoplanet discoveries.

---

## ⚙️ **How It Works — System Overview**

### 1. **Integrated Data Pipeline**

We built a structured pipeline that collects, cleans, and harmonizes NASA mission datasets.
The process standardizes column names, fixes unit inconsistencies, and handles missing values — transforming raw archives into consistent, machine-readable CSVs.

### 2. **Feature Engineering & Mapping**

A dedicated feature-engineering module computes derived astrophysical features such as:

* logarithmic orbital period, radius, and depth,
* stellar effective temperature, surface gravity, and luminosity,
* and binary indicators for missing data.

A **mapping layer** ensures full compatibility across KOI, TOI, and TESS, enabling a single model to operate on multiple missions.

### 3. **Machine-Learning Core**

We trained and compared several supervised classifiers — **Random Forest**, **Logistic Regression**, and **Gradient Boosting (XGBoost)** — using the **Kepler Object of Interest (KOI)** dataset.
Performance was validated with **ROC-AUC**, **F1-score**, and **precision-recall**, and class balancing was optimized using **SMOTE** oversampling.
All trained models, encoders, and metrics are stored as reusable artifacts for reproducibility.

### 4. **Inference & Automation**

The trained model can process new **TESS/TOI** data in two modes:

* **Batch prediction** (CSV input for large-scale datasets)
* **Single-object prediction** (JSON input for real-time analysis)

The output includes predicted class labels and probability scores (`CONFIRMED`, `CANDIDATE`, `FALSE POSITIVE`).

### 5. **Interactive Frontend & 3D Visualization**

We developed an intuitive **Streamlit web application** that connects directly to the ML model.
Users can:

* Upload data or input parameters manually
* View class probabilities and feature transformations
* Adjust the model’s decision threshold
* Explore a **3D interactive planet** visualization rendered via **Three.js**, where radius and color dynamically respond to stellar temperature and planetary size

This visual layer bridges science and user experience — making the discovery process transparent and engaging.

### 6. **Backend Simulation**

A lightweight **FastAPI backend** serves as a mock prediction service with endpoints for `/predict_one` and `/predict_batch`.
It allows real-time integration between the ML system and the frontend, simulating a deployable API for research environments.

---

## 🧩 **Project Architecture**

| Layer              | Description                     | Key Components                                                   |
| ------------------ | ------------------------------- | ---------------------------------------------------------------- |
| **Data Layer**     | NASA raw and cleaned datasets   | `data/raw`, `data/cleaned`                                       |
| **ML Core**        | Training, evaluation, inference | `ml/train_supervised.py`, `ml/evaluate.py`, `ml/infer_tess.py`   |
| **Research Layer** | Experiments and EDA notebooks   | `notebooks/EDA.ipynb`, `feature_eng.ipynb`, `ensembleTOI.ipynb`  |
| **Backend**        | Mock REST API for predictions   | `backend/mock_api.py`                                            |
| **Frontend**       | Streamlit dashboard + 3D viewer | `frontend/app_streamlit.py`, `planet_3d.py`, `exoplanet_3d.html` |

---

## 🌠 **End-to-End Workflow**

1. **Data ingestion** → NASA archives → `/data/raw`
2. **Cleaning & mapping** → harmonized datasets → `/data/cleaned`
3. **Feature engineering** → computed astrophysical features
4. **Model training & evaluation** → supervised ML + ensemble analysis
5. **Inference on new data** → predictions for TOI/TESS objects
6. **Visualization & UI** → interactive dashboard + 3D simulation
7. **API simulation** → mock backend endpoints for deployment testing

---

## 💡 **Benefits and Impact**

* **Accelerates discovery:** AI replaces manual classification, enabling thousands of signals to be analyzed in seconds.
* **Enhances transparency:** All models, datasets, and notebooks are open and reproducible.
* **Bridges science and design:** Combines astrophysics, data analytics, and interactive visualization in one cohesive tool.
* **Educational value:** The intuitive 3D interface makes space science accessible for both researchers and students.

---

## 🧰 **Tools and Technologies**

| Category          | Technologies Used                                                    |
| ----------------- | -------------------------------------------------------------------- |
| **Languages**     | Python (pandas, numpy, scikit-learn, xgboost), JavaScript (Three.js) |
| **Frameworks**    | Streamlit, FastAPI                                                   |
| **ML Libraries**  | scikit-learn, imbalanced-learn, joblib                               |
| **Visualization** | Matplotlib, Three.js, Streamlit components                           |
| **Data Handling** | pandas, NumPy                                                        |
| **File Storage**  | CSV datasets, JSON model artifacts                                   |
| **Environments**  | Jupyter Notebooks for research, Streamlit for deployment             |

---

## 🌌 **Creativity and Design Factors**

* We unified real NASA data from different missions into one interoperable format — something rarely achieved in open exoplanet research.
* The integration of **AI classification + scientific visualization + UI interactivity** creates a tool that is both functional for scientists and engaging for non-experts.
* The **3D planet viewer**, coded from scratch in Three.js, visually connects numerical predictions to physical characteristics — a blend of creativity and data science.
* Our design prioritizes **explainability, transparency, and accessibility** — enabling others to extend or retrain the model easily.

---

## 👩‍🚀 **Team Mission Statement**

> *“We believe that artificial intelligence can extend humanity’s ability to explore the universe.
> Our mission is to make exoplanet discovery faster, more accurate, and open to everyone.”*

---

✅ **In summary:**
Our project transforms raw astrophysical archives into an intelligent, interpretable AI system that not only predicts new exoplanets — but also lets you **see them in 3D**.
It’s a bridge between data science and cosmic curiosity — a step toward the future of **AI-assisted space exploration.**
