# India City Livability Index & Karnataka State Deep-Dive

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75.svg)](https://plotly.com/)
[![Tests](https://img.shields.io/badge/Tests-49%2F49%20Passing-brightgreen.svg)]()

> **Academic Project — Master of Computer Applications (MCA)**  
> **Institution:** BMS Institute of Technology & Management (BMSITM), Bengaluru  
> **Student:** Ruchira Ramesh Yallur (`1BY25MC086`), II Sem MCA  
> **Faculty Guide:** Dr. Shivakumara T, Associate Professor, Department of MCA  

---

## 📌 Executive Summary
Urban livability is inherently subjective: a student prioritizes top-tier colleges and affordability; a family values safety, air quality, and hospitals; a retiree values healthcare access and climate comfort; a remote worker looks for high-speed connectivity and low living costs.

The **India City Livability Index** is an interactive, data-driven web platform that enables citizens to customize city rankings using **real-time priority sliders (1–10)** across **7 core livability dimensions**. The platform features a **two-tier hierarchical architecture**:
1. **Pan-India Macro Scope:** Benchmarks 30 major metropolitan and Tier-2 cities across India.
2. **Karnataka Regional Deep-Dive:** Provides granular district-level analysis across all 31 districts of Karnataka using official GIS boundaries.

The system incorporates unsupervised machine learning (**PCA** and **K-Means Clustering**) to discover latent development patterns and group cities into natural tiers without human bias.

---

## 🏛️ The 7 Core Livability Pillars

| # | Pillar | Metric Description | Directionality | Data Source |
|---|---|---|---|---|
| 1 | **Climate** | Annual climate comfort score & temperature index | Higher is better | OpenWeatherMap / IMD |
| 2 | **Air Quality** | Annual mean particulate matter ($PM_{2.5}$) | Lower is better | CPCB (`city_day.csv`) / KSPCB |
| 3 | **Safety** | Total cognizable IPC crimes per 100,000 population | Lower is better | NCRB (`district_crimes_2014.csv`) / KSP |
| 4 | **Education** | Higher education quality & NIRF institution density | Higher is better | NIRF Rankings / Census / AISHE |
| 5 | **Affordability** | Estimated monthly cost of living (INR) | Lower is better | Numbeo India / Open City |
| 6 | **Healthcare** | Hospital beds and health facilities per lakh | Higher is better | National Health Portal (`data.gov.in`) / DES Karnataka |
| 7 | **Connectivity** | Airport, metro, rail junction, and digital access index | Higher is better | TRAI / Transit Open Data |

---

## 📐 Mathematical Formulation

### 1. Unified 0–100 MinMax Normalization
To compare disparate metrics fairly, every raw metric is normalized onto a standardized $0 \text{--} 100$ scale:

* **For "Higher is Better" metrics:**
  $$\text{Score} = \left(\frac{x - x_{\min}}{x_{\max} - x_{\min}}\right) \times 100$$

* **For "Lower is Better" metrics (inverted):**
  $$\text{Score} = 100 - \left[\left(\frac{x - x_{\min}}{x_{\max} - x_{\min}}\right) \times 100\right]$$

### 2. Weighted Composite Index
$$\text{Livability Index} = \frac{\sum_{i=1}^7 (\text{Score}_i \times \text{Weight}_i)}{\sum_{i=1}^7 \text{Weight}_i}$$

Where $\text{Weight}_i \in [1, 10]$ are dynamically controlled by the user via interactive sliders or 1-click persona presets.

---

## 🔬 Machine Learning Analytics

### 1. Principal Component Analysis (PCA)
* Features standardized using `StandardScaler` ($\mu = 0, \sigma^2 = 1$).
* Reduces the 7 dimensions to 2 principal components for 2D visualization:
  * **Pan-India:** $PC_1 = 48.98\%$, $PC_2 = 34.13\%$ ($\mathbf{83.11\%}$ total variance captured).
  * **Karnataka:** $PC_1 = 52.89\%$, $PC_2 = 33.80\%$ ($\mathbf{86.69\%}$ total variance captured).
* **Loadings:** Identifies $PC_1$ as the *Development & Infrastructure Gradient* and $PC_2$ as the *Environmental Wellness & Safety Gradient*.

### 2. K-Means Clustering & Archetype Labeling
* Evaluated optimal $k$ across range $[2, 6]$ using **Silhouette Score** and **Inertia**.
* Segregates cities/districts into 4 natural archetypes:
  * **Tier 1 Metro Powerhouse** (e.g. Bengaluru Urban, Mumbai, Delhi)
  * **Affordable High-Growth Hub** (e.g. Udupi, Mangaluru, Mysuru, Coimbatore)
  * **Wellness & Quality-of-Life Haven** (e.g. Kodagu, Chikkamagaluru, Shivamogga, Shimla)
  * **Developing Regional / Interior Belt** (e.g. Kalaburagi, Raichur, Bidar)

---

## 🖥️ Streamlit Web Dashboard Features

1. **🗺️ Rankings & Live Map:**
   * Dynamic Pan-India Bubble Map & Karnataka District Polygon Map.
   * Interactive Leaderboard with Progress Bars, Rank Shifts ($+3$, $-2$), and CSV export.
2. **⚖️ Head-to-Head Comparison:**
   * Side-by-side selection of any two cities/districts.
   * Overlaid dual-trace Radar Chart with color-coded area fills.
   * Delta table showing pillar-by-pillar differences and winner tags.
3. **🔬 Machine Learning Clusters:**
   * 2D PCA interactive scatter plot colored by cluster archetype.
   * Centroid profiles across all 7 dimensions.
4. **🏙️ Deep-Dive Profile:**
   * Individual city inspection card with radar chart and progress bars benchmarked against state/national averages.

---

## 📁 Project Architecture

```text
india live/
├── data/
│   ├── raw/
│   │   ├── raw_national_cities.csv       # 30 Pan-India major metros
│   │   ├── raw_karnataka_districts.csv   # 31 Karnataka districts
│   │   ├── city_day.csv                  # Real CPCB air quality dataset
│   │   ├── district_crimes_2014.csv      # Real NCRB district crime dataset
│   │   └── top_500_indian_cities.csv     # Real Census demographic dataset
│   ├── processed/
│   │   ├── master_national.csv           # Cleaned, normalized Pan-India master
│   │   └── master_karnataka.csv          # Cleaned, normalized Karnataka master
│   ├── karnataka.geojson                 # Official GIS district boundaries
│   └── DATA_SOURCES.md                   # Real dataset documentation & links
├── src/
│   ├── __init__.py
│   ├── data_loader.py                    # Module 1: Ingestion & schema validation
│   ├── preprocessor.py                   # Module 2: Normalization (0-100) & imputation
│   ├── calculator.py                     # Module 3: Weighted scoring & personas
│   ├── ml_engine.py                      # Module 4: PCA & K-Means clustering
│   └── plots.py                          # Module 5: Plotly maps, radar & scatter charts
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   ├── test_modules_1_and_2_comprehensive.py
│   ├── test_calculator.py
│   ├── test_ml_engine.py
│   ├── test_plots.py
│   └── test_app.py
├── app.py                                # Module 6: Streamlit multi-page web app
├── requirements.txt                      # Module 7: Python dependencies
├── .gitignore                            # Clean repository filters
└── README.md                             # Documentation
```

---

## 🚀 Quick Start Guide

### 1. Clone & Set Up Environment
```powershell
# Navigate to project folder
cd "c:\Users\USER\Desktop\india live"

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full Test Suite (49 Tests)
```powershell
python -m pytest tests/
```

### 3. Launch the Interactive Dashboard
```powershell
python -m streamlit run app.py
```
Open your browser and navigate to: `http://localhost:8501`

---

## 🌐 UN Sustainable Development Goals (SDG) Alignment
* **SDG 11 (Sustainable Cities and Communities):** Equips citizens and policymakers with transparent livability metrics.
* **SDG 3 (Good Health and Well-Being):** Surfaces environmental pollution ($PM_{2.5}$) and healthcare infrastructure disparities.
* **SDG 9 (Industry, Innovation and Infrastructure):** Evaluates public transit maturity and digital connectivity.
* **SDG 10 (Reduced Inequalities):** Highlights regional disparities between metropolitan capitals and tier-2/tier-3 interior districts.
* **SDG 17 (Partnerships for the Goals):** Built entirely on open government data ecosystems (CPCB, NCRB, data.gov.in, OpenCity).
