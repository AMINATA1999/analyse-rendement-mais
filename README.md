# 🌽 Analyse statistique et prédiction du rendement du maïs

Projet réalisé dans le cadre du Master Data Science (UPEC), module *Analyse et Traitement de Données Numériques*.

**Problème métier** : une exploitation agricole veut comprendre comment le type de sol, l'engrais, les précipitations, la température et la surface cultivée influencent le rendement du maïs, afin d'optimiser ses pratiques.

## 🎯 Résultats clés

| Modèle | MAE | RMSE | R² |
|---|---|---|---|
| Régression linéaire | 2,07 | 2,43 | −0,003 |
| Forêt aléatoire | 2,06 | 2,50 | −0,06 |
| LightGBM | 2,14 | 2,56 | −0,11 |
| XGBoost | 2,21 | 2,68 | −0,22 |

**Conclusion principale** : aucune des variables disponibles n'explique le rendement. Les corrélations avec le rendement sont proches de 0, et aucun modèle ne fait mieux qu'une prédiction par la moyenne (R² ≤ 0). Les modèles les plus complexes (XGBoost, LightGBM) font même moins bien que la régression linéaire, signe qu'ils apprennent du bruit.

## 📊 Données

1 000 parcelles décrites par 6 variables : surface (ha), type de sol (argileux, limoneux, sableux), engrais (kg/ha), précipitations (mm), température (°C) et rendement (t/ha, variable cible).

## 🛠️ Démarche

1. **Formulation du problème** : variable cible, variables explicatives, question métier.
2. **Statistiques descriptives** du rendement : moyenne (7,34 t/ha), médiane, mode, variance, écart-type, étendue ; histogramme et boxplot.
3. **Corrélations** : matrice de corrélation et heatmap.
4. **Test statistique** : ANOVA pour tester l'effet du type de sol sur le rendement.
5. **Modélisation** : comparaison de 4 modèles de régression (régression linéaire, forêt aléatoire, XGBoost, LightGBM) avec MAE, RMSE et R².

## 🔍 Analyse critique et limites

- **Pas de signal exploitable** : les rendements moyens par type de sol sont très proches (7,3 à 7,6 t/ha), et les corrélations sont toutes inférieures à 0,1 en valeur absolue.
- **Évaluation** : un seul découpage entraînement / test (80 / 20) ; une validation croisée donnerait des résultats plus stables.
- **Encodage du type de sol** : il est codé de 1 à 3, ce qui impose un ordre artificiel aux catégories. Un encodage one-hot serait plus adapté pour la régression linéaire.
- **Pistes** : ajouter des variables plus explicatives (humidité du sol, historique des parcelles, variétés de maïs), qui sont sans doute les vrais déterminants du rendement.

## 🧰 Outils

Python · pandas · NumPy · SciPy · scikit-learn · XGBoost · LightGBM · Matplotlib · seaborn

## 📁 Fichiers

```
├── analyse_rendement_mais.py   # Script complet
├── rapport_analyse.pdf         # Rapport d'analyse
└── rendement_mais.csv          # Données
```

## 👩‍💻 Autrice

**Ndeye Aminata Gadiaga**
[LinkedIn](https://linkedin.com/in/ndeye-aminata-gadiaga-26a97725b) · [GitHub](https://github.com/AMINATA1999)
