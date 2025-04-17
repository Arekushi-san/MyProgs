#Exercice Panda

import pandas as pd

# Init des données
data = {
    'Nom': ['Alice', 'Bob', 'Charlie', 'David', 'Eva', 'Frank', 'Gina'],
    'Âge': [25, 30, 35, 40, 28, 32, 27],
    'Ville': ['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Nice', 'Paris', 'Lyon'],
    'Score': [88, 92, 95, 70, 85, 77, 90]
}

dataf = pd.DataFrame(data)

# Sauvegarde fichier csv
dataf.to_csv('personnes.csv', index=False)
print("Fichier 'personnes.csv' sauvegardé.")

# Lecture fichier csv
dataf = pd.read_csv('personnes.csv')
print("\nDonnées chargées depuis le CSV :\n", dataf)

# Ajout colonne "Catégorie", dépend du score (excelent si >90, bon si >80, sinon moyen)
dataf['Catégorie'] = dataf['Score'].apply(lambda s: 'Excellent' if s >= 90 else ('Bon' if s >= 80 else 'Moyen'))

# Tri par score décroissant
dataf_sorted = dataf.sort_values(by='Score', ascending=False)
print("\nDonnées triées par score décroissant :\n", dataf_sorted)

# Regroupement par ville & moyenne des scores
grouped = dataf.groupby('Ville')['Score'].mean().reset_index()
print("\nMoyenne des scores par ville :\n", grouped)

# Filtrage (ne guarde que les "Excellent")
dataf_excellents = dataf[dataf['Catégorie'] == 'Excellent']
print("\nPersonnes avec un score 'Excellent' :\n", dataf_excellents)

# sauvegarde de ce dataframe dans fichier csv
dataf_excellents.to_csv('excellents.csv', index=False)
print("\nFichier 'excellents.csv' exporté.")
