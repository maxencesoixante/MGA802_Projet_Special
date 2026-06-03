import pandas as pd


def charger_donnees_thermocouples(chemin_fichier):
    """
    Charge les données des thermocouples depuis un fichier Excel (.xlsx).
    Ignore le TC1 et retourne un DataFrame propre.
    """
    # Utilisation de read_excel (skiprows=4 permet de sauter les lignes d'en-tête comme Tmax, Tmin)
    # Si vos données commencent à une autre ligne, ajustez le chiffre 4.
    df = pd.read_excel(chemin_fichier, skiprows=4)

    # Liste des colonnes qui nous intéressent (on ignore TC1)
    colonnes_utiles = ['Time (s)', 'TC2 (C)', 'TC3 (C)', 'TC4 (C)', 'TC5 (C)']

    # Filtrage et suppression des lignes où il manque des données (NaN)
    df_propre = df[colonnes_utiles].dropna()

    return df_propre


# Fonction utilitaire pour le jumeau numérique
def extraire_temperatures_a_t(df, temps_cible):
    """
    Trouve la ligne correspondant au temps_cible et retourne un dictionnaire
    avec les températures des 4 thermocouples.
    """
    # Trouve l'index du temps le plus proche de temps_cible
    idx_proche = (df['Time (s)'] - temps_cible).abs().idxmin()
    ligne = df.loc[idx_proche]

    # Retourne un dictionnaire prêt pour le calibrage
    return {
        'TC2': ligne['TC2 (C)'],
        'TC3': ligne['TC3 (C)'],
        'TC4': ligne['TC4 (C)'],
        'TC5': ligne['TC5 (C)']
    }