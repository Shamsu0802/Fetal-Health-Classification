## API Implementation for Random Forest Model

### Introduction

To make the fetal health prediction model more accessible and practical, an API has been developed that serves the trained **Random Forest** model. This API allows external applications, websites, or users to submit fetal data in real time and receive accurate health status predictions instantly. By exposing the model through an API, the project enables easy integration into healthcare software systems, facilitating timely decision-making and improving prenatal care.

---

### API Overview

- **Framework Used:** Flask (Python web framework)
- **Endpoint:** `/predict`
- **HTTP Method:** POST
- **Input Format:** JSON containing fetal health features
- **Output Format:** JSON containing predicted class and confidence score

---

### How It Works

Clients send a POST request with a JSON payload containing the required fetal health features — such as baseline fetal heart rate, accelerations, decelerations, variability measures, and other cardiotocogram indicators. The API processes the input data, applies the trained Random Forest model, and returns a prediction class indicating whether the fetal health is Normal, Suspect, or Pathological, along with the model’s confidence in that prediction.

---

### Sample Request

```bash
POST /predict
Content-Type: application/json

{
  "baseline_value": 140,
  "accelerations": 2,
  "fetal_movement": 3,
  "uterine_contractions": 0,
  "light_decelerations": 1,
  "severe_decelerations": 0,
  "prolongued_decelerations": 0,
  "abnormal_short_term_variability": 25,
  "mean_value_of_short_term_variability": 8,
  "percentage_of_time_with_abnormal_long_term_variability": 20,
  "mean_value_of_long_term_variability": 10,
  "histogram_width": 50,
  "histogram_min": 100,
  "histogram_max": 160,
  "histogram_number_of_peaks": 4,
  "histogram_number_of_zeroes": 0,
  "histogram_mode": 140,
  "histogram_mean": 140,
  "histogram_median": 140,
  "histogram_variance": 200,
  "histogram_tendency": 0.5
}
