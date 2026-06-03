"""
visualization.py
----------------
Fonctions de visualisation pour le Jumeau Numérique MGA 802.

Contient :
  - afficher_champ_temperature()  : carte de chaleur 2D (inchangée)
  - tracer_comparaison_tc()       : ★ NOUVELLE — courbes simulées vs réelles
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from skfem.visuals.matplotlib import plot as skfem_plot


# ─────────────────────────────────────────────────────────────────────
# 1. Carte de chaleur 2D (fonction existante, légèrement améliorée)
# ─────────────────────────────────────────────────────────────────────

def afficher_champ_temperature(mesh, T_field,
                                titre="Distribution de température à l'interface (t final)"):
    """
    Affiche la carte de chaleur 2D du champ nodal calculé par scikit-fem.
    Marqueurs colorés aux positions des 4 thermocouples.
    """
    coords_tc = {
        'TC2': ( 0.000,  0.000),
        'TC3': ( 0.000,  0.060),
        'TC4': ( 0.020, -0.060),
        'TC5': (-0.020, -0.060),
    }
    couleurs = {'TC2': 'white', 'TC3': 'red', 'TC4': 'green', 'TC5': 'blue'}

    fig, ax = plt.subplots(figsize=(6, 9))
    skfem_plot(mesh, T_field, ax=ax, shading='gouraud', colorbar=True)

    for nom, (x, y) in coords_tc.items():
        ax.plot(x, y, 'o', color=couleurs[nom], markersize=8,
                markeredgecolor='black', label=nom)

    ax.set_title(titre, fontsize=12, pad=10)
    ax.set_xlabel("Largeur X (m)")
    ax.set_ylabel("Longueur Y (m)")
    ax.legend(loc='upper right')
    plt.tight_layout()
    return fig          # On retourne la figure pour pouvoir la sauvegarder


# ─────────────────────────────────────────────────────────────────────
# 2. ★ NOUVELLE : comparaison temporelle simulé vs réel
# ─────────────────────────────────────────────────────────────────────

def tracer_comparaison_tc(historique_simul, df_experimental,
                           colonne_temps='Temps_s',
                           noms_tc=('TC2', 'TC3', 'TC4', 'TC5'),
                           titre_global="Jumeau Numérique — Simulation vs Expérience"):
    """
    Trace l'évolution temporelle de chaque TC (simulé ET expérimental)
    sur 4 sous-graphiques indépendants partageant l'axe des temps.

    Paramètres
    ----------
    historique_simul : dict
        Retourné par JumeauNumeriqueInduction.simuler().
        Doit contenir les clés 'temps', 'TC2', 'TC3', 'TC4', 'TC5'.

    df_experimental : pd.DataFrame
        DataFrame Pandas lu depuis le CSV.
        Doit contenir une colonne temps (colonne_temps) et une colonne
        par TC nommée identiquement ('TC2', 'TC3', 'TC4', 'TC5').
        Exemple de lecture :
            df = pd.read_csv("5TC_226A.csv", sep=';', decimal=',')

    colonne_temps : str
        Nom de la colonne temps dans df_experimental.

    noms_tc : tuple[str]
        Noms des thermocouples à tracer (doit correspondre aux deux sources).

    titre_global : str
        Titre affiché en haut de la figure.

    Retour
    ------
    fig : matplotlib.figure.Figure

    Pourquoi 4 sous-graphiques ?
    ----------------------------
    Les TC ont des plages de température très différentes (TC2 ≈ 250 °C
    contre TC4/TC5 ≈ 30 °C). Les regrouper sur un même axe Y rendrait
    les courbes froides illisibles. Les sous-graphiques résolvent cela.
    """

    # ── Extraction des données simulées depuis l'historique ───────────
    # historique['temps'] est une liste Python → on la convertit en
    # tableau NumPy pour faciliter les opérations (cohérence avec df).
    temps_simul = np.array(historique_simul['temps'])

    # ── Mise en page : 4 lignes, 1 colonne, hauteurs égales ──────────
    # GridSpec permet des ajustements fins (hspace) impossibles avec
    # plt.subplots() seul.
    fig = plt.figure(figsize=(10, 11))
    fig.suptitle(titre_global, fontsize=14, fontweight='bold', y=1.01)

    gs = gridspec.GridSpec(nrows=len(noms_tc), ncols=1,
                           hspace=0.45,      # espace vertical entre sous-graphs
                           figure=fig)

    couleurs_simul = {'TC2': '#e63946', 'TC3': '#2a9d8f',
                      'TC4': '#457b9d', 'TC5': '#f4a261'}
    couleurs_exp   = {'TC2': '#c1121f', 'TC3': '#168975',
                      'TC4': '#1d3557', 'TC5': '#e76f51'}

    axes = []   # on garde une référence aux axes pour un éventuel post-traitement

    for i, nom_tc in enumerate(noms_tc):
        ax = fig.add_subplot(gs[i])
        axes.append(ax)

        # ── Courbe simulée ────────────────────────────────────────────
        temp_simul_tc = np.array(historique_simul[nom_tc])
        ax.plot(temps_simul, temp_simul_tc,
                color=couleurs_simul[nom_tc],
                linewidth=2.0,
                linestyle='-',
                label=f"{nom_tc} — Simulé (JN)")

        # ── Courbe expérimentale (depuis le DataFrame Pandas) ─────────
        # On vérifie que la colonne existe avant de tracer pour éviter
        # un KeyError silencieux lors des démos.
        if nom_tc in df_experimental.columns and colonne_temps in df_experimental.columns:
            temps_exp = df_experimental[colonne_temps].to_numpy()
            temp_exp  = df_experimental[nom_tc].to_numpy()

            ax.plot(temps_exp, temp_exp,
                    color=couleurs_exp[nom_tc],
                    linewidth=1.5,
                    linestyle='--',
                    marker='o',
                    markersize=3,
                    label=f"{nom_tc} — Expérimental")

            # ── Calcul de l'erreur RMS sur la plage commune ───────────
            # On interpole la simulation aux instants expérimentaux pour
            # comparer des valeurs au même t (les pas dt peuvent différer).
            t_min = max(temps_simul[0], temps_exp[0])
            t_max = min(temps_simul[-1], temps_exp[-1])
            masque = (temps_exp >= t_min) & (temps_exp <= t_max)

            if masque.sum() > 1:
                temp_simul_interp = np.interp(temps_exp[masque],
                                              temps_simul, temp_simul_tc)
                rms = np.sqrt(np.mean((temp_simul_interp - temp_exp[masque])**2))
                ax.set_title(f"{nom_tc}   (RMS = {rms:.2f} °C)", fontsize=10)
            else:
                ax.set_title(nom_tc, fontsize=10)
        else:
            ax.set_title(f"{nom_tc}  (données expérimentales manquantes)", fontsize=10)

        # ── Mise en forme de l'axe ────────────────────────────────────
        ax.set_ylabel("Température (°C)", fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(fontsize=8, loc='upper left')

        # L'axe X n'est étiqueté que sur le graphique du bas
        if i < len(noms_tc) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Temps (s)", fontsize=10)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────
# Exemple autonome (test sans digital_twin.py)
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Historique fictif pour tester la visualisation sans lancer la simulation
    temps_test   = list(range(0, 11))          # 0 à 10 secondes
    historique_test = {
        'temps': temps_test,
        'TC2': [22 + 23 * t for t in temps_test],
        'TC3': [22 + 2  * t for t in temps_test],
        'TC4': [22 + 0.8* t for t in temps_test],
        'TC5': [22 + 0.8* t for t in temps_test],
        'T_final': None,
    }

    # DataFrame expérimental fictif (normalement lu depuis le CSV)
    df_test = pd.DataFrame({
        'Temps_s': [0, 2, 4, 6, 8, 10],
        'TC2':     [22, 68, 115, 162, 208, 252],
        'TC3':     [22, 24, 27,  32,  37,  42 ],
        'TC4':     [22, 23, 24,  26,  28,  30 ],
        'TC5':     [22, 23, 24,  26,  28,  30 ],
    })

    fig = tracer_comparaison_tc(historique_test, df_test)
    plt.show()