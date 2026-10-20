import pandas as pd
from sklearn.model_selection import train_test_split

# On charge les donnees brutes
df = pd.read_csv("data/raw_data/raw.csv")

# On ne garde que les colonnes numeriques (retire la colonne de date)
df = df.select_dtypes(include="number")

# La cible est la derniere colonne : silica_concentrate
X = df.drop(columns=["silica_concentrate"])
y = df["silica_concentrate"]

# Separation en ensemble d'entrainement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Sauvegarde des 4 datasets dans data/processed_data
X_train.to_csv("data/processed_data/X_train.csv", index=False)
X_test.to_csv("data/processed_data/X_test.csv", index=False)
y_train.to_csv("data/processed_data/y_train.csv", index=False)
y_test.to_csv("data/processed_data/y_test.csv", index=False)

print("Split OK")
