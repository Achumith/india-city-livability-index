# Real Data Sources & Download Directory for Livability Index

This document provides direct links, file descriptions, and column mappings for real open-source and government datasets covering the 7 Livability Pillars for both **Pan-India Metros** and **Karnataka Districts**.

---

## 1. Air Quality (PM2.5 & AQI)
* **Dataset Name:** Air Quality Data in India (Central Pollution Control Board - CPCB)
* **Source:** Kaggle / CPCB
* **Direct Download Link:** [Kaggle Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
* **File to Download:** `city_day.csv` (contains daily PM2.5, PM10, AQI for major Indian cities including Bengaluru, Delhi, Mumbai, Hyderabad, Chennai, etc.)
* **Official Government Portal:** [CPCB Central Ambient Air Quality Portal](https://app.cpcbccr.com/AQI_India/)
* **Karnataka Specific:** [Karnataka State Pollution Control Board (KSPCB)](https://kspcb.karnataka.gov.in)
* **Key Columns Used:** `City`, `Date`, `PM2.5`, `AQI`

---

## 2. Safety & Crime Statistics
* **Dataset Name:** District-Wise Crimes in India (NCRB)
* **Source:** Kaggle / National Crime Records Bureau (NCRB)
* **Direct Download Link:** [Kaggle Crime in India Dataset](https://www.kaggle.com/datasets/rajanand/crime-in-india)
* **File to Download:** `01_District_wise_crimes_committed_IPC_2001_2012.csv` (or modern releases: `Crime in India 2022`)
* **Official Government Portal:** [NCRB Crime in India Official Portal](https://ncrb.gov.in/)
* **Karnataka Specific:** [Karnataka State Police Statistics](https://ksp.karnataka.gov.in/page/Statistics/en)
* **Key Columns Used:** `STATE/UT`, `DISTRICT`, `TOTAL IPC CRIMES`, `POPULATION` (to derive Crime Rate per Lakh)

---

## 3. Affordability & Cost of Living
* **Dataset Name:** Numbeo Cost of Living Index by City / India Living Cost
* **Source:** Numbeo / Kaggle
* **Direct Download Link (Kaggle):** [Cost of Living Index by City - Numbeo](https://www.kaggle.com/datasets/prasertk/cost-of-living-index-by-country-and-city)
* **Live Web Portal:** [Numbeo Cost of Living India](https://www.numbeo.com/cost-of-living/country_result.jsp?country=India)
* **Key Columns Used:** `City`, `Cost of Living Index`, `Rent Index`, `Groceries Index` (or monthly estimated expenses)

---

## 4. Education (Higher Education & NIRF)
* **Dataset Name:** National Institutional Ranking Framework (NIRF)
* **Official Government Portal:** [NIRF India Official Portal](https://www.nirfindia.org/)
* **Supplementary Source:** [All India Survey on Higher Education (AISHE)](https://aishe.gov.in)
* **What to Extract:** Count of ranked institutions in Top 100 per city/district (Universities, Engineering, Medical, Management, Colleges).

---

## 5. Healthcare (Hospitals & Medical Infrastructure)
* **Dataset Name:** Government Hospitals and Beds Across Indian States and Districts
* **Source:** Open Government Data Platform India (`data.gov.in`)
* **Direct Portal Link:** [data.gov.in - State/UT/District Hospital Beds](https://data.gov.in/resource/state-ut-wise-number-government-hospitals-and-beds-rural-and-urban-areas)
* **Karnataka District Deep-Dive:** [Karnataka Directorate of Economics & Statistics (DES)](https://des.karnataka.gov.in)
  * Download the annual *"District at a Glance"* PDF/Excel reports for each district (e.g. Mysuru, Kalaburagi, Dakshina Kannada) which report the exact number of District Hospitals, Taluk Hospitals, Community Health Centres (CHCs), Primary Health Centres (PHCs), and bed counts.

---

## 6. Connectivity (Transit, Rail, Air, Telecom)
* **Dataset Name:** AAI Airport Traffic & TRAI Telecom Subscription Data
* **Airports:** [Airports Authority of India (AAI) Traffic Data](https://www.aai.aero/en/business-opportunities/aai-traffic-data)
* **Telecom/Broadband:** [Telecom Regulatory Authority of India (TRAI)](https://www.trai.gov.in)
* **Urban Transit:** Open data on Namma Metro (BMRCL), Delhi Metro (DMRC), Mumbai Metro, and Indian Railways Division headquarters.

---

## 7. Climate & Weather
* **API Service:** [OpenWeatherMap API](https://openweathermap.org/api) (Free Tier gives 1,000 API calls/day)
* **Official Historical Weather:** [India Meteorological Department (IMD) - Climatological Normals](https://mausam.imd.gov.in)
* **Karnataka Specific:** [Karnataka State Natural Disaster Monitoring Centre (KSNDMC)](http://ksndmc.org/) for district rainfall and weather reports.

---

## 8. Geospatial Boundary Files (GeoJSON)
* **Karnataka Districts GeoJSON:** Successfully downloaded to `data/karnataka.geojson` (Contains 30+ district polygon boundaries for Plotly interactive choropleth maps).
* **India States GeoJSON:** Available via community open data repositories (`datameet/maps`).
