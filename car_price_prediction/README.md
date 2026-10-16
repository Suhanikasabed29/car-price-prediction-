# Car Price Prediction — Full Stack Implementation

A complete frontend + backend machine-learning project for predicting used-car selling prices from the supplied `car data.csv` dataset.

## Stack

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Flask REST API
- **Machine Learning:** scikit-learn Random Forest Regressor
- **Data:** `data/car data.csv`

## Project structure

```text
car_price_prediction/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── car data.csv
├── models/
│   └── car_price_model.pkl
└── static/
    ├── index.html
    ├── style.css
    └── app.js
```

## Run locally

### 1. Create and activate a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train/retrain the model

The ZIP already contains a trained model, but you can retrain it from the included dataset:

```bash
python train_model.py
```

### 4. Start the backend

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## API

### Health check

`GET /api/health`

### Prediction

`POST /api/predict`

Example JSON:

```json
{
  "Car_Name": "swift",
  "Year": 2017,
  "Present_Price": 6.5,
  "Driven_kms": 30000,
  "Fuel_Type": "Petrol",
  "Selling_type": "Dealer",
  "Transmission": "Manual",
  "Owner": 0
}
```

Example response:

```json
{
  "predicted_price": 5.42,
  "currency": "lakh INR",
  "message": "Prediction generated successfully."
}
```

## Notes

- Prices are represented in **Indian lakh rupees**, matching the dataset's scale.
- `Car_Name` is one-hot encoded and unknown names are accepted by the model because the encoder uses `handle_unknown="ignore"`.
- The backend validates numeric inputs before prediction.
- The model is saved with pickle after training.
