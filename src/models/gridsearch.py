import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# Donnees d'entrainement normalisees
X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").values.ravel()

# Modele de regression + grille de parametres a tester
model = GradientBoostingRegressor(random_state=42)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
}

grid = GridSearchCV(model, param_grid, cv=3, scoring="r2")
grid.fit(X_train, y_train)

# Sauvegarde des meilleurs parametres dans models/
with open("models/best_params.pkl", "wb") as f:
    pickle.dump(grid.best_params_, f)

print("Meilleurs parametres :", grid.best_params_)
