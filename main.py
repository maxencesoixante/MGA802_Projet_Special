"""
main.py — Point d'entrée du Jumeau Numérique MGA 802
=====================================================
Chaîne complète :
  1. Lecture de config.yaml
  2. Chargement des données Excel
  3. Calibration multi-paramètres (Q_ind, h_eff)
  4. Simulation avec historique complet (TC + champs 2D)
  5. Visualisation comparative 1D
  6. ★ Boucle interactive : carte thermique 2D à la demande
"""

import pathlib
import yaml
import numpy as np
import matplotlib
matplotlib.use('TkAgg')          # backend interactif (PyCharm / terminal)
import matplotlib.pyplot as plt

from src.data_loader   import charger_donnees_thermocouples
from src.digital_twin  import JumeauNumeriqueInduction
from src.visualization import afficher_champ_temperature, tracer_comparaison_tc


# ═════════════════════════════════════════════════════════════════════
# 1. LECTURE DE LA CONFIGURATION
# ═════════════════════════════════════════════════════════════════════
# yaml.safe_load() parse le YAML en un dict Python natif.
# pathlib.Path garantit la portabilité Windows / macOS / Linux.
CHEMIN_CONFIG = pathlib.Path("config.yaml")

print("=" * 60)
print("  JUMEAU NUMÉRIQUE — MGA 802")
print("=" * 60)
print(f"\n[0/5] Lecture de la configuration : {CHEMIN_CONFIG}")

with open(CHEMIN_CONFIG, encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Lecture des chemins et paramètres numériques depuis le dict config
CHEMIN_EXCEL = pathlib.Path(config['fichiers']['donnees_excel'])
DT           = float(config['simulation']['dt'])
T_INITIALE   = float(config['simulation']['T_initiale'])

print(f"      dt = {DT} s  |  T_init = {T_INITIALE} °C")


# ═════════════════════════════════════════════════════════════════════
# 2. CHARGEMENT DES DONNÉES EXPÉRIMENTALES
# ═════════════════════════════════════════════════════════════════════
print(f"\n[1/5] Chargement des données : {CHEMIN_EXCEL}")
df = charger_donnees_thermocouples(str(CHEMIN_EXCEL))
print(f"      {len(df)} lignes  |  "
      f"t = {df['Temps_s'].min():.1f} → {df['Temps_s'].max():.1f} s")

# Détection automatique de l'instant d'arrêt = pic de TC2
idx_pic          = df['TC2'].idxmax()
temps_arret_auto = float(df.loc[idx_pic, 'Temps_s'])
temps_max_exp    = float(df['Temps_s'].max())
print(f"      Pic TC2 détecté à t = {temps_arret_auto} s  "
      f"(T_max = {df.loc[idx_pic, 'TC2']:.1f} °C)")


# ═════════════════════════════════════════════════════════════════════
# 3. INITIALISATION DU JUMEAU NUMÉRIQUE
# ═════════════════════════════════════════════════════════════════════
# La classe reçoit le dict config — aucune valeur en dur dans le code.
print("\n[2/5] Initialisation du jumeau numérique …")
jumeau = JumeauNumeriqueInduction(config)
print(f"      Maillage {config['maillage']['nx']}×{config['maillage']['ny']} | "
      f"N_nœuds = {jumeau.basis.N}")

# Estimation de la mémoire pour les champs 2D
N_pas   = int(temps_max_exp / DT)
mem_mo  = (N_pas + 1) * jumeau.basis.N * 8 / 1e6
print(f"      Mémoire estimée pour champs_2d : {mem_mo:.1f} Mo "
      f"({N_pas+1} pas × {jumeau.basis.N} nœuds × 8 o)")


# ═════════════════════════════════════════════════════════════════════
# 4. CALIBRATION MULTI-PARAMÈTRES
# ═════════════════════════════════════════════════════════════════════
print("\n[3/5] Calibration multi-paramètres (Q_ind, h_eff) …")

historique_tc2_exp = {
    'temps': df['Temps_s'].tolist(),
    'TC2'  : df['TC2'].tolist(),
}

amplitude_optimale, h_eff_optimal = jumeau.calibrer_modele(
    historique_tc2_experimental=historique_tc2_exp,
    temps_arret=temps_arret_auto,
    temps_cible=temps_max_exp,
)


# ═════════════════════════════════════════════════════════════════════
# 5. SIMULATION FINALE AVEC HISTORIQUE COMPLET
# ═════════════════════════════════════════════════════════════════════
print(f"\n[4/5] Simulation finale : 0 → {temps_max_exp:.1f} s …")
historique = jumeau.simuler(
    amplitude_q=amplitude_optimale,
    h_eff=h_eff_optimal,
    temps_cible=temps_max_exp,
    temps_arret=temps_arret_auto,
)
print(f"      {len(historique['temps'])} pas enregistrés. "
      f"champs_2d : {historique['champs_2d'].shape}")


# ═════════════════════════════════════════════════════════════════════
# 6. VISUALISATION 1D — comparaison simulé vs expérimental
# ═════════════════════════════════════════════════════════════════════
print("\n[5/5] Génération des figures comparatives …")

fig_tc = tracer_comparaison_tc(
    historique_simul=historique,
    df_experimental=df,
    titre_global=(f"Jumeau Numérique — Simulé vs Expérimental\n"
                  f"Q = {amplitude_optimale:.2e} W/m³  |  "
                  f"h_eff = {h_eff_optimal:.0f} W/m³·K"),
)
fig_tc.savefig(config['fichiers']['sortie_comparaison'],
               dpi=150, bbox_inches='tight')
print(f"      Sauvegardé : {config['fichiers']['sortie_comparaison']}")

# Affichage non-bloquant — la boucle interactive peut s'ouvrir juste après
plt.show(block=False)
plt.pause(0.5)


# ═════════════════════════════════════════════════════════════════════
# 7. ★ BOUCLE INTERACTIVE — carte thermique 2D à la demande
# ═════════════════════════════════════════════════════════════════════
def _trouver_indice_temps(temps_array: np.ndarray, t_demande: float) -> int:
    """
    Retourne l'indice i tel que temps_array[i] est le plus proche de t_demande.

    Utilise np.argmin sur les distances absolues — O(N_pas), acceptable
    car appelé seulement lors d'une saisie utilisateur (pas dans une boucle
    de calcul). Retourne un int Python pour l'indexation de champs_2d.
    """
    return int(np.argmin(np.abs(temps_array - t_demande)))


print("\n" + "─" * 60)
print("  MODE INTERACTIF — Visualisation des champs thermiques 2D")
print("─" * 60)
print(f"  Plage disponible : t = 0.0 → {temps_max_exp:.1f} s")
print("  Tapez un instant en secondes, ou 'q' pour quitter.\n")

temps_array = historique['temps']   # np.ndarray — accès rapide

while True:
    saisie = input("  ▶ Instant t (s) ou 'q' : ").strip()

    # ── Condition de sortie ───────────────────────────────────────
    if saisie.lower() in ('q', 'quit', 'exit', ''):
        print("  Fermeture du programme. Au revoir !")
        break

    # ── Validation de la saisie ───────────────────────────────────
    # On tente la conversion float ; si l'utilisateur tape "abc",
    # ValueError est attrapée proprement sans crasher le programme.
    try:
        t_demande = float(saisie.replace(',', '.'))   # tolère "10,5" et "10.5"
    except ValueError:
        print(f"  ✗ '{saisie}' n'est pas un nombre valide. Réessayez.\n")
        continue

    # ── Vérification de la plage ──────────────────────────────────
    if not (temps_array[0] <= t_demande <= temps_array[-1]):
        print(f"  ✗ t = {t_demande} s est hors plage "
              f"[{temps_array[0]:.1f}, {temps_array[-1]:.1f}] s.\n")
        continue

    # ── Récupération du champ 2D ──────────────────────────────────
    # L'indexation directe champs_2d[i] est O(1) grâce à la pré-allocation.
    idx      = _trouver_indice_temps(temps_array, t_demande)
    t_reel   = float(temps_array[idx])
    T_champ  = historique['champs_2d'][idx]   # vue sur la ligne i, pas de copie

    T_min = T_champ.min()
    T_max = T_champ.max()
    print(f"  ✓ Affichage à t = {t_reel:.1f} s  "
          f"(T_min = {T_min:.1f} °C,  T_max = {T_max:.1f} °C)")

    fig_2d = afficher_champ_temperature(
        mesh  = jumeau.mesh,
        T_field = T_champ,
        titre = (f"Champ de température — t = {t_reel:.1f} s\n"
                 f"Q = {amplitude_optimale:.2e} W/m³  |  "
                 f"h_eff = {h_eff_optimal:.0f} W/m³·K"),
    )
    plt.show(block=False)
    plt.pause(0.1)
    print()

# Maintien des fenêtres ouvertes après la sortie de la boucle
plt.show()