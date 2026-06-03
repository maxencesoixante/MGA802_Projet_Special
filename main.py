from src.data_loader import charger_donnees_thermocouples, extraire_temperatures_a_t
from src.digital_twin import JumeauNumeriqueInduction
from src.visualization import afficher_champ_temperature


def main():
    print("--- Démarrage du Jumeau Numérique (Projet MGA 802) ---")

    # 1. Chargement des données expérimentales
    chemin_excel = "data/5TC_226A.xlsx"
    print(f"Chargement des données depuis {chemin_excel}...")
    df = charger_donnees_thermocouples(chemin_excel)

    # Choix du pas de temps à analyser (ex: t = 10.0 secondes, près du pic)
    temps_analyse = 10.0
    temperatures_cibles = extraire_temperatures_a_t(df, temps_cible=temps_analyse)
    print(f"Températures cibles à t={temps_analyse}s : {temperatures_cibles}")

    # 2. Initialisation du modèle éléments finis
    print("Initialisation du maillage et du modèle physique...")
    jumeau = JumeauNumeriqueInduction()

    # 3. Étape de Calibration (Le cœur du Jumeau Numérique)
    # L'algorithme va chercher la puissance optimale de la bobine
    puissance_optimale = jumeau.calibrer_modele(temperatures_cibles)

    # 4. Simulation finale avec la puissance trouvée
    print("Simulation du champ 2D avec la puissance calibrée...")
    T_finale = jumeau.simuler(amplitude_q=puissance_optimale, temps_cible=temps_analyse)

    # 5. Affichage des résultats !
    afficher_champ_temperature(jumeau.mesh, T_finale, titre=f"Jumeau Numérique à t={temps_analyse}s")


if __name__ == "__main__":
    main()