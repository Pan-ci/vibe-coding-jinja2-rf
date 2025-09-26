from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import joblib
import sys
from app.dependencies.unik import (
    genders, years, cgpas, marital_statuses, courses
) # or without ()
from app.services.predict_service import run_prediction

app = FastAPI()


root = Path(__file__).parent.parent.resolve()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

best_model_path = root / 'app' / 'models' / 'best_model.pkl'
nested_cv_path = root / 'app' / 'models' / 'nested_cv.pkl'
templates_path = root / 'app' / 'templates'
static_path = root / 'app' / 'static'
icon_path = root / 'app' / 'static' / 'favicon.ico'

templates = Jinja2Templates(directory=templates_path)
app.mount("/static", StaticFiles(directory=static_path), name="static")

models = {
    "best_model": joblib.load(best_model_path),
    "nested_cv": joblib.load(nested_cv_path)
}

@app.get("/favicon.ico")
async def favicon():
    return RedirectResponse(url=icon_path)

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    model_keys = list(models.keys())

    return templates.TemplateResponse("index.html", {
        "request": request,
        "default_age": 20,
        "courses": courses,
        "genders": genders,
        "years": years,
        "cgpas": cgpas,
        "maritals": marital_statuses,
        "model_keys": model_keys
    })

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    name: str = Form(...),
    course: str = Form(...),
    year: str = Form(...),
    cgpa: str = Form(...),
    gender: str = Form(...),
    marital: str = Form(...),
    age: int = Form(...),
    model_choice: str = Form(...)
):
    
    model = models[model_choice] 
    manual_input = {
        'Choose your gender': gender,
        'Age': age,
        'What is your course?': course,
        'Your current year of Study': year,
        'What is your CGPA?': cgpa,
        'Marital status': marital
    }

    input = {
        'Name': name,
        'Choose your gender': gender,
        'Age': age,
        'What is your course?': course,
        'Your current year of Study': year,
        'What is your CGPA?': cgpa,
        'Marital status': marital
    }

    prediction, proba = run_prediction(model, manual_input)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "prediction": "Yes" if prediction == 1 else "No",
        "proba_yes": f"{proba[1]*100:.2f}%",
        "proba_no": f"{proba[0]*100:.2f}%",
        "manual_input": input,
        'model_choice': model_choice
    })
