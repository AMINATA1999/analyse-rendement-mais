import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mode, f_oneway
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Chargement des données
url = "rendement_mais.csv"
data = pd.read_csv(url,sep=',',on_bad_lines='skip')  
print(data.head())
print("Colonnes du fichier:", data.columns)


# Encodage de la colonne 'TYPE_SOL'
type_sol_mapping = {
    'Argileux': 1,
    'Sableux': 2,
    'Limoneux': 3
}

data['TYPE_SOL_ENCODED'] = data['TYPE_SOL'].map(type_sol_mapping)

#Afficher les premières lignes pour vérifier
print("\n",data.head())

# Mesures de tendance centrale
mean_rendement = data['RENDEMENT_T_HA'].mean()
median_rendement = data['RENDEMENT_T_HA'].median()
mode_rendement = mode(data['RENDEMENT_T_HA'], keepdims=True).mode[0]

# Mesures de dispersion
variance_rendement = data['RENDEMENT_T_HA'].var()
stdev_rendement = data['RENDEMENT_T_HA'].std()
range_rendement = data['RENDEMENT_T_HA'].max() - data['RENDEMENT_T_HA'].min()

# Calcul de la matrice de corrélation
correlation_matrix = data.drop(columns=['TYPE_SOL']).corr()

# Affichage de la heatmap des corrélations
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap des corrélations")
plt.show()

# Résultats statistiques
print("resultat statistiques \n")
print({
    "Moyenne": mean_rendement,
    "Médiane": median_rendement,
    "Mode": mode_rendement,
    "Variance": variance_rendement,
    "Écart-type": stdev_rendement,
    "Étendue": range_rendement
})

# Visualisation des données
plt.figure(figsize=(10, 5))
sns.histplot(data['RENDEMENT_T_HA'], bins=5, kde=True)
plt.title("Histogramme du rendement")
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x=data['RENDEMENT_T_HA'])
plt.title("Boxplot du rendement")
plt.show()

# Analyse de la variance (ANOVA) sur le type de sol
groups = [data['RENDEMENT_T_HA'][data['TYPE_SOL'] == t] for t in data['TYPE_SOL'].unique()]
anova_test = f_oneway(*groups)
p_value = anova_test.pvalue
print("P_Value=",p_value)


# Étape 4 : Modélisation
X = data[['SURFACE_HA', 'ENGRAIS_KG_HA', 'PRECIPITATIONS_MM', 'TEMPERATURE_C', 'TYPE_SOL_ENCODED']]
y = data['RENDEMENT_T_HA']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Résultats finaux
print("\n resulat finaux")
descriptive_stats = {
    
    "p-value ANOVA": p_value,
    "MAE": mae,
    "RMSE": rmse,
    "R²": r2
}

print(descriptive_stats)

from sklearn.ensemble import RandomForestRegressor

model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

# Évaluation
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print("R² (Random Forest) :", r2_rf)
print("Mae (Random Forest) :", mae_rf)
print("Rmse (Random Forest) :", rmse_rf)


from xgboost import XGBRegressor

model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)

r2_xgb = r2_score(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print("R² (XGBoost) :", r2_xgb)
print("Mae² (XGBoost) :", mae_xgb)
print("RMSE² (XGBoost) :", rmse_xgb)

import lightgbm as lgb

# Création du modèle LightGBM
model_lgb = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Entraînement du modèle
model_lgb.fit(X_train, y_train)

# Prédiction
y_pred_lgb = model_lgb.predict(X_test)

# Évaluation du modèle
mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
r2_lgb = r2_score(y_test, y_pred_lgb)

# Affichage des résultats
print("\n Résultats de LightGBM:")
print(f" MAE : {mae_lgb:.4f}")
print(f" RMSE : {rmse_lgb:.4f}")
print(f" R² : {r2_lgb:.4f}")
import lightgbm as lgb

# Création du modèle LightGBM
model_lgb = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Entraînement du modèle
model_lgb.fit(X_train, y_train)

# Prédiction
y_pred_lgb = model_lgb.predict(X_test)

# Évaluation du modèle
mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
r2_lgb = r2_score(y_test, y_pred_lgb)

# Affichage des résultats
print("\n Résultats de LightGBM:")
print(f"🔹 MAE : {mae_lgb:.4f}")
print(f"🔹 RMSE : {rmse_lgb:.4f}")
print(f"🔹 R² : {r2_lgb:.4f}")

