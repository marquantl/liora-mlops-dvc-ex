import pandas as pd
from sklearn.preprocessing import StandardScaler

# On charge les ensembles d'entrainement et de test
X_train = pd.read_csv("data/processed_data/X_train.csv")
X_test = pd.read_csv("data/processed_data/X_test.csv")

# On apprend la normalisation sur le train, on l'applique aux deux
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sauvegarde des donnees normalisees dans data/processed_data
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(
    "data/processed_data/X_train_scaled.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(
    "data/processed_data/X_test_scaled.csv", index=False)

print("Normalisation OK")
