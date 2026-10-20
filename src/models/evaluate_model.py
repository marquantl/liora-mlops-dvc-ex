import json
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score

# Donnees de test normalisees
X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv").values.ravel()

# On recupere le modele entraine
with open("models/gbr_model.pkl", "rb") as f:
    model = pickle.load(f)

# Predictions
predictions = model.predict(X_test)

# On sauvegarde les predictions dans data/
pd.DataFrame({"prediction": predictions}).to_csv("data/prediction.csv", index=False)

# On calcule les metriques et on les sauvegarde dans metrics/scores.json
scores = {
    "mse": mean_squared_error(y_test, predictions),
    "r2": r2_score(y_test, predictions),
}
with open("metrics/scores.json", "w") as f:
    json.dump(scores, f, indent=4)

print("Scores :", scores)
