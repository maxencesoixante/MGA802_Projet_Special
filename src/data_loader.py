"""
data_loader.py
--------------
Chargement et préparation des données expérimentales depuis le fichier Excel.
"""

import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# Noms de colonnes normalisés (utilisés dans tout le projet)
# ─────────────────────────────────────────────────────────────────────
# On centralise ici la correspondance entre les noms Excel bruts et les
# noms courts utilisés dans le reste du code (digital_twin, visualization).
# Si le fichier Excel change de format, on ne modifie QUE ce dictionnaire.

COLONNES_EXCEL = {
    'Time (s)': 'Temps_s',   # colonne temps
    'TC2 (C)' : 'TC2',
    'TC3 (C)' : 'TC3',
    'TC4 (C)' : 'TC4',
    'TC5 (C)' : 'TC5',
}


def charger_donnees_thermocouples(chemin_fichier):
    """
    Charge les données des thermocouples depuis un fichier Excel (.xlsx).
    Ignore le TC1 et retourne un DataFrame propre avec des noms de colonnes
    normalisés ('Temps_s', 'TC2', 'TC3', 'TC4', 'TC5').

    Paramètres
    ----------
    chemin_fichier : str | Path
        Chemin vers le fichier .xlsx (ex. "5TC_226A.xlsx")

    Retour
    ------
    df_propre : pd.DataFrame
        Colonnes : Temps_s, TC2, TC3, TC4, TC5
        Lignes   : une par pas de temps, sans NaN.
    """
    # skiprows=4 saute les lignes d'en-tête (Tmax, Tmin, unités…)
    df_brut = pd.read_excel(chemin_fichier, skiprows=4)

    # On ne garde que les colonnes utiles (TC1 ignoré)
    colonnes_brutes = list(COLONNES_EXCEL.keys())           # noms Excel
    df_filtre = df_brut[colonnes_brutes].dropna().copy()

    # Renommage vers les noms courts du projet
    df_propre = df_filtre.rename(columns=COLONNES_EXCEL)

    # Remise à zéro de l'index après dropna (bonne pratique)
    df_propre = df_propre.reset_index(drop=True)

    return df_propre


def extraire_temperatures_a_t(df, temps_cible):
    """
    Retourne les températures des 4 TC à l'instant le plus proche de temps_cible.

    Paramètres
    ----------
    df          : pd.DataFrame retourné par charger_donnees_thermocouples()
    temps_cible : float — instant désiré en secondes

    Retour
    ------
    dict : {'TC2': float, 'TC3': float, 'TC4': float, 'TC5': float}
          Prêt à être passé directement à calibrer_modele().
    """
    idx_proche = (df['Temps_s'] - temps_cible).abs().idxmin()
    ligne = df.loc[idx_proche]

    return {
        'TC2': float(ligne['TC2']),
        'TC3': float(ligne['TC3']),
        'TC4': float(ligne['TC4']),
        'TC5': float(ligne['TC5']),
    }


# ─────────────────────────────────────────────────────────────────────
# Test rapide (optionnel)
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    chemin = sys.argv[1] if len(sys.argv) > 1 else "/Users/maxencedubois/PycharmProjects/MGA802_Jumeau_Numerique/data/5TC_226A.xlsx"

    df = charger_donnees_thermocouples(chemin)
    print("Colonnes :", df.columns.tolist())
    print("Premières lignes :\n", df.head())
    print("\nTemps min/max :", df['Temps_s'].min(), "→", df['Temps_s'].max(), "s")

    t_cible = df['Temps_s'].iloc[-1]   # dernier instant du fichier
    print(f"\nTempératures à t={t_cible:.1f} s :", extraire_temperatures_a_t(df, t_cible))