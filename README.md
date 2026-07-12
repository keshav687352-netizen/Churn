# BonTelco Customer Churn Prediction App

A full-stack machine learning application that predicts customer churn using a trained XGBoost model and exposes a live FastAPI backend for inference.

## Architecture

- Frontend: React + Vite
- Backend: FastAPI
- Model: serialized XGBoost pipeline in `frontend/models/customer_churn_model.pkl`

## Features

- Customer churn prediction via `/predict`
- Health check via `/health`
- Probability outputs for stay and churn
- React UI connected to live backend JSON
- Validation and error handling

## Local Run

Backend:

```bash
python -m uvicorn frontend.api:app --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

## API

- `GET /health`
- `POST /predict`

Example request body:

```json
{
  "Age": 22,
  "Tenure": 25,
  "Usage Frequency": 14,
  "Support Calls": 4,
  "Payment Delay": 27,
  "Contract Length": "Monthly",
  "Total Spend": 598,
  "Last Interaction": 9
}
```

## Notes

This repo currently keeps the working runtime stable while exposing a cleaner `backend/` layout shim for future restructuring.
