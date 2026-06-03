"""
main.py — Point d'entrée du Jumeau Numérique MGA 802
-----------------------------------------------------
Chaîne complète :
  1. Chargement des données Excel           (data_loader.py)
  2. Calibration de la puissance de bobine  (digital_twin.py)
  3. Simulation avec historique complet     (digital_twin.py)
  4. Visualisation comparative              (visualization.py)
"""

import matplotlib.pyplot as plt

from src.data_loader  import charger_donnees_thermocouples, extraire_temperatures_a_t
from src.digital_twin import JumeauNumeriqueInduction
from src.visualization import afficher_champ_temperature, tracer_comparaison_tc


# ─────────────────────────────────────────────────────────────────────
# 0. Paramètres globaux
# ─────────────────────────────────────────────────────────────────────
CHEMIN_EXCEL  = "data/5TC_226A.xlsx"   # ← adapte si nécessaire
TEMPS_CIBLE   = 10.0              # secondes — instant de calibration
DT            = 1.0               # pas de temps [s]
T_INITIALE    = 22.0              # °C — température ambiante initiale


# ─────────────────────────────────────────────────────────────────────
# 1. Chargement des données expérimentales
# ─────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  JUMEAU NUMÉRIQUE — MGA 802")
print("=" * 55)

print(f"\n[1/4] Chargement de : {CHEMIN_EXCEL}")
df = charger_donnees_thermocouples(CHEMIN_EXCEL)
print(f"      {len(df)} lignes chargées  |  "
      f"t = {df['Temps_s'].min():.1f} → {df['Temps_s'].max():.1f} s")


# ─────────────────────────────────────────────────────────────────────
# 2. Calibration du modèle
# ─────────────────────────────────────────────────────────────────────
print(f"\n[2/4] Calibration à t = {TEMPS_CIBLE} s …")

# Extrait les températures réelles au temps de calibration
t_reelles = extraire_temperatures_a_t(df, TEMPS_CIBLE)
print(f"      Températures expérimentales : {t_reelles}")

jumeau = JumeauNumeriqueInduction()
amplitude_optimale = jumeau.calibrer_modele(
    vraies_temperatures_a_t_cible=t_reelles,
    temps_cible=TEMPS_CIBLE,
    dt=DT,
)


# ─────────────────────────────────────────────────────────────────────
# 3. Simulation complète avec historique temporel
# ─────────────────────────────────────────────────────────────────────
# On utilise le temps maximal du fichier Excel pour couvrir toute
# la plage expérimentale, pas seulement l'instant de calibration.
temps_max_exp = float(df['Temps_s'].max())

print(f"\n[3/4] Simulation de 0 → {temps_max_exp:.1f} s  (dt = {DT} s) …")
historique = jumeau.simuler(
    amplitude_q=amplitude_optimale,
    temps_cible=temps_max_exp,
    dt=DT,
    T_init=T_INITIALE,
)
print(f"      {len(historique['temps'])} pas enregistrés.")


# ─────────────────────────────────────────────────────────────────────
# 4. Visualisation
# ─────────────────────────────────────────────────────────────────────
print("\n[4/4] Génération des figures …")

# Figure A — comparaison temporelle simulé vs expérimental
fig_tc = tracer_comparaison_tc(
    historique_simul=historique,
    df_experimental=df,
    titre_global=(f"Jumeau Numérique — Simulé vs Expérimental\n"
                  f"(Q_bobine = {amplitude_optimale:.2e} W/m³)"),
)

# Figure B — champ de température 2D au dernier instant
fig_2d = afficher_champ_temperature(
    mesh=jumeau.mesh,
    T_field=historique['T_final'],
    titre=f"Champ de température à t = {temps_max_exp:.0f} s",
)

# Sauvegarde
fig_tc.savefig("comparaison_tc.png", dpi=150, bbox_inches='tight')
fig_2d.savefig("champ_2D.png",       dpi=150, bbox_inches='tight')
print("      Figures sauvegardées : comparaison_tc.png  |  champ_2D.png")

plt.show()
print("\nTerminé.")