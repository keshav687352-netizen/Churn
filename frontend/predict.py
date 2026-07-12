from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "customer_churn_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

age = int(input("Enter Age: "))
tenure = int(input("Enter Tenure: "))
usage = int(input("Enter Usage Frequency: "))
support = int(input("Enter Support Calls: "))
payment = int(input("Enter Payment Delay: "))
contract = input("Enter Contract Length (Monthly/Quarterly/Annual): ").strip().title()
spend = int(input("Enter Total Spend: "))
interaction = int(input("Enter Last Interaction: "))

new_customer = pd.DataFrame(
    {
        "Age": [age],
        "Tenure": [tenure],
        "Usage Frequency": [usage],
        "Support Calls": [support],
        "Payment Delay": [payment],
        "Contract Length": [contract],
        "Total Spend": [spend],
        "Last Interaction": [interaction],
    }
)

new_customer["Contract Length"] = pd.Categorical(
    new_customer["Contract Length"],
    categories=["Monthly", "Quarterly", "Annual"],
)

prediction = model.predict(new_customer)[0]
probability = model.predict_proba(new_customer)[0]

if prediction == 1:
    print("\n⚠ Customer is likely to CHURN.")
    print(f"Confidence : {probability[1] * 100:.2f}%")
else:
    print("\n✅ Customer is likely to STAY.")
    print(f"Confidence : {probability[0] * 100:.2f}%")