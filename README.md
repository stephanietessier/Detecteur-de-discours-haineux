# Détecteur de discours haineux en ligne — IA responsable

Projet éducatif prêt pour GitHub et Google Colab. Il propose un détecteur simple de messages problématiques en ligne : discours haineux, harcèlement, insultes, menace ou message acceptable.

> Important : ce projet est un prototype pédagogique. Il ne doit pas remplacer une modération humaine, ni servir à surveiller des personnes sans consentement. Les décisions sensibles doivent toujours être revues par une personne.

## Objectifs

- Classer des messages courts selon leur niveau de risque.
- Fournir une explication simple des mots qui influencent la décision.
- Proposer une file de révision humaine pour les cas incertains.
- Respecter la vie privée : pas d'envoi externe, données fictives, traitement local.

## Structure

```text
detecteur_discours_haineux/
├── app/
│   └── streamlit_app.py
├── data/
│   └── messages_moderation_fictifs.csv
├── docs/
│   └── fiche_ethique.md
├── notebooks/
│   └── Detecteur_Discours_Haineux_Colab.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── moderation_model.py
│   └── utils.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Lancer sur Google Colab

1. Ouvrir `notebooks/Detecteur_Discours_Haineux_Colab.ipynb`.
2. Exécuter les cellules dans l'ordre.
3. Tester des messages dans la cellule de prédiction.

## Lancer localement

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Catégories

- `acceptable` : message non problématique.
- `insulte` : attaque verbale ou mépris.
- `harcelement` : intimidation, pression répétée ou humiliation.
- `haine` : attaque visant un groupe protégé ou une identité.
- `menace` : intention violente ou intimidation grave.

## Approche IA

Le modèle utilise `TfidfVectorizer` + `LogisticRegression`, deux méthodes classiques, rapides et explicables. Le but est de montrer un pipeline complet : données, entraînement, prédiction, score de confiance, explication et interface.

## Limites

- Les données sont fictives et petites.
- Le modèle peut se tromper, surtout avec l'ironie, les citations, les dialectes, le contexte culturel ou les messages ambigus.
- Il peut contenir des biais selon les exemples d'entraînement.
- Il faut ajouter un vrai protocole d'audit avant tout usage public.

## Améliorations possibles

- Ajouter davantage de données annotées par des humains.
- Ajouter une classe `à réviser` lorsque la confiance est faible.
- Mesurer les faux positifs et faux négatifs par catégorie.
- Mettre en place un journal d'audit anonymisé.
- Ajouter une politique d'appel pour les personnes modérées.

## Licence

MIT — à adapter selon ton usage.
