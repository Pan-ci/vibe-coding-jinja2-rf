import pandas as pd

def run_prediction(model, user_input: dict):
    '''
    prediksi dan kembalikan nilai prediksi dan praobabilitas
    '''
    df = pd.DataFrame([user_input])
    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0]

    return prediction, proba