from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
import pandas as pd
from src.predict_pipeline import load_artifacts, predict_churn

app = FastAPI()
templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../templates")
)

model, scaler, training_columns = load_artifacts()

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    gender: str = Form(...),
    SeniorCitizen: int = Form(0),
    Partner: str = Form("No"),
    Dependents: str = Form("No"),
    tenure: int = Form(0),
    PhoneService: str = Form("No"),
    PaperlessBilling: str = Form("No"),
    MonthlyCharges: float = Form(0),
    TotalCharges: float = Form(0),
    MultipleLines: str = Form("No"),
    InternetService: str = Form("No"),
    OnlineSecurity: str = Form("No"),
    OnlineBackup: str = Form("No"),
    DeviceProtection: str = Form("No"),
    TechSupport: str = Form("No"),
    StreamingTV: str = Form("No"),
    StreamingMovies: str = Form("No"),
    Contract: str = Form("Month-to-month"),
    PaymentMethod: str = Form("Electronic check"),
):
    data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "PaperlessBilling": PaperlessBilling,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaymentMethod": PaymentMethod,
    }

    df = pd.DataFrame([data])
    preds, probs = predict_churn(df, model, scaler, training_columns)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": int(preds[0][0]),
            "probability": float(probs[0][0]),
        },
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", host="0.0.0.0", port=5000, reload=False)

