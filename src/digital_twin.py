import pandas as pd
import numpy as np
from skfem import *
from scipy.optimize import minimize


class JumeauNumeriqueInduction:
    def __init__(self):
        # --- 1. Propriétés thermophysiques (PEKK-FC / Carbone) ---
        self.rho = 1310.0  # Densité [kg/m^3]
        self.cp = 959.0  # Capacité thermique [J/kg.K]

        # ⚠️ Fibres alignées avec la longueur (axe Y)
        self.kx = 0.658  # Conductivité transverse [W/m.K]
        self.ky = 6.8  # Conductivité longitudinale [W/m.K]

        # --- 2. Création du maillage 2D centré en (0,0) ---
        # X va de -0.020 à 0.020 (40 mm)
        # Y va de -0.060 à 0.060 (120 mm)
        self.mesh = MeshTri.init_tensor(
            np.linspace(-0.020, 0.020, 20),
            np.linspace(-0.060, 0.060, 60)
        )
        self.basis = Basis(self.mesh, ElementTriP1())

        # Coordonnées des Thermocouples (en mètres)
        self.coords_tc = {
            'TC2': (0.0, 0.0),
            'TC3': (0.0, 0.060),
            'TC4': (0.020, -0.060),
            'TC5': (-0.020, -0.060)
        }

    def formuler_probleme(self, dt):
        """Formulation variationnelle (Galerkin) pour scikit-fem"""

        @BilinearForm
        def a(u, v, w):
            terme_masse = (self.rho * self.cp / dt) * u * v
            terme_conduction = self.kx * u.grad[0] * v.grad[0] + self.ky * u.grad[1] * v.grad[1]
            return terme_masse + terme_conduction

        @LinearForm
        def L(v, w):
            chaleur_precedente = (self.rho * self.cp / dt) * w.T_old * v
            source_induction = w.Q_ind * v
            return chaleur_precedente + source_induction

        return a, L

    def generer_source_induction(self, amplitude, rayon=0.015):
        """Modélise la bobine comme une tache gaussienne centrée en (0,0)"""
        x = self.basis.doflocs[0]
        y = self.basis.doflocs[1]
        return amplitude * np.exp(-(x ** 2 + y ** 2) / (2 * rayon ** 2))

    def extraire_temperature_tc(self, T_field, x_tc, y_tc):
        """Trouve le nœud le plus proche des coordonnées du TC et retourne sa température"""
        x_mesh = self.mesh.p[0]
        y_mesh = self.mesh.p[1]
        distances_carrées = (x_mesh - x_tc) ** 2 + (y_mesh - y_tc) ** 2
        id_noeud_proche = np.argmin(distances_carrées)
        return T_field[id_noeud_proche]

    def simuler(self, amplitude_q, temps_cible=10.0, dt=1.0):
        """Fait tourner la simulation jusqu'à un temps donné"""
        a, L = self.formuler_probleme(dt)
        A = asm(a, self.basis)

        # Température initiale (ex: 22°C d'après votre fichier)
        T_actuelle = np.full(self.basis.N, 22.0)
        Q_ind = self.generer_source_induction(amplitude_q)

        # Boucle de temps
        for t in np.arange(dt, temps_cible + dt, dt):
            b = asm(L, self.basis, T_old=T_actuelle, Q_ind=Q_ind)
            T_actuelle = solve(A, b)

        return T_actuelle

    def calibrer_modele(self, vraies_temperatures_a_t10):
        """
        Trouve la puissance de la bobine (amplitude_q) qui minimise l'erreur
        entre la simulation et les 4 thermocouples (Le concept de Digital Twin)
        """

        def fonction_erreur(amplitude_test):
            amplitude_test = amplitude_test[0]  # scipy envoie un array

            # 1. On lance la simulation avec cette amplitude
            T_simul = self.simuler(amplitude_q=amplitude_test, temps_cible=10.0)

            # 2. On extrait les températures aux 4 TC
            erreur_totale = 0
            for nom_tc, coord in self.coords_tc.items():
                t_simul_tc = self.extraire_temperature_tc(T_simul, coord[0], coord[1])
                t_reel_tc = vraies_temperatures_a_t10[nom_tc]

                # Erreur quadratique
                erreur_totale += (t_simul_tc - t_reel_tc) ** 2

            return erreur_totale

        # On lance l'algorithme d'optimisation (valeur initiale arbitraire : 1e6 W/m³)
        print("Démarrage du calibrage du jumeau numérique...")
        resultat = minimize(fonction_erreur, x0=[1e6], method='Nelder-Mead')
        print(f"Calibrage terminé ! Amplitude optimale trouvée : {resultat.x[0]:.2e} W/m³")

        return resultat.x[0]


# ==========================================
# Exemple d'utilisation dans votre `main.py`
# ==========================================
if __name__ == "__main__":
    jumeau = JumeauNumeriqueInduction()

    # Imaginons qu'à t=10 secondes dans votre CSV "5TC_226A", vous ayez ces températures :
    temperatures_reelles_t10 = {
        'TC2': 250.0,
        'TC3': 40.0,
        'TC4': 30.0,
        'TC5': 30.0
    }

    # 1. Le modèle trouve tout seul la puissance de la bobine !
    puissance_optimale = jumeau.calibrer_modele(temperatures_reelles_t10)

    # 2. Vous pouvez maintenant relancer la simulation avec cette puissance
    # pour extraire et tracer la distribution 2D complète.
    T_finale = jumeau.simuler(amplitude_q=puissance_optimale, temps_cible=10.0)

    # 3. Pour visualiser avec matplotlib (Optionnel mais recommandé pour la démo MGA)
    from skfem.visuals.matplotlib import plot
    import matplotlib.pyplot as plt

    plot(jumeau.mesh, T_finale, shading='gouraud', colorbar=True)
    plt.title("Distribution de température à l'interface (Digital Twin)")
    plt.show()