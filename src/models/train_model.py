import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor

# Donnees d'entrainement normalisees
X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").values.ravel()

# On recupere les meilleurs parametres trouves par le GridSearch
with open("models/best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

# Entrainement du modele avec ces parametres
model = GradientBoostingRegressor(random_state=42, **best_params)
model.fit(X_train, y_train)

# Sauvegarde du modele entraine dans models/
with open("models/gbr_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Modele entraine et sauvegarde")
