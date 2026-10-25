import pickle

model_file = 'pipeline.bin'

with open(model_file, 'rb') as f_in:
    artifact = pickle.load(f_in)

lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41
}

if isinstance(artifact, tuple):
    dv, model = artifact
    X = dv.transform([lead])
    prob = model.predict_proba(X)[0, 1]
else:
    prob = artifact.predict_proba([lead])[0, 1]

print(f"Conversion probability: {prob:.3f}")