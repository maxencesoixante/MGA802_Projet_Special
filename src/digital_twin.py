import numpy as np
import pandas as pd
from skfem import *
from scipy.optimize import minimize


class JumeauNumeriqueInduction:
    def __init__(self):
        # --- 1. Propriétés thermophysiques (PEKK-FC / Carbone) ---
        self.rho = 1310.0       # Densité [kg/m^3]
        self.cp  = 959.0        # Capacité thermique [J/kg.K]

        # ⚠️ Fibres alignées avec la longueur (axe Y)
        self.kx = 0.658         # Conductivité transverse [W/m.K]
        self.ky = 6.8           # Conductivité longitudinale [W/m.K]

        # --- 2. Création du maillage 2D centré en (0,0) ---
        self.mesh = MeshTri.init_tensor(
            np.linspace(-0.020,  0.020, 20),
            np.linspace(-0.060,  0.060, 60)
        )
        self.basis = Basis(self.mesh, ElementTriP1())

        # --- 3. Coordonnées des thermocouples (en mètres) ---
        self.coords_tc = {
            'TC2': ( 0.000,  0.000),
            'TC3': ( 0.000,  0.060),
            'TC4': ( 0.020, -0.060),
            'TC5': (-0.020, -0.060),
        }

        # --- 4. Pré-calcul des indices de nœuds les plus proches de chaque TC ---
        # Fait une seule fois ici pour ne pas recalculer à chaque pas de temps.
        # C'est une optimisation importante : avec N pas de temps, on économise
        # 4 * N recherches de minimum.
        self._indices_noeuds_tc = {
            nom: self._trouver_noeud_proche(x, y)
            for nom, (x, y) in self.coords_tc.items()
        }

    # ------------------------------------------------------------------
    # Méthodes internes (privées, préfixe "_")
    # ------------------------------------------------------------------

    def _trouver_noeud_proche(self, x_tc, y_tc):
        """Retourne l'indice du nœud du maillage le plus proche de (x_tc, y_tc)."""
        x_mesh = self.mesh.p[0]
        y_mesh = self.mesh.p[1]
        distances_sq = (x_mesh - x_tc)**2 + (y_mesh - y_tc)**2
        return int(np.argmin(distances_sq))

    def _extraire_temperatures_tc(self, T_field):
        """
        Extrait la température aux 4 TC depuis un champ nodal T_field.

        Retourne un dictionnaire  {'TC2': float, 'TC3': float, ...}
        au lieu d'une valeur unique : cela permet de tout collecter en une
        seule passe sur les 4 TC, sans appels répétés.
        """
        return {
            nom: float(T_field[idx])
            for nom, idx in self._indices_noeuds_tc.items()
        }

    # ------------------------------------------------------------------
    # Formulation variationnelle (inchangée)
    # ------------------------------------------------------------------

    def formuler_probleme(self, dt):
        """Formulation Galerkin (Euler implicite en temps) pour scikit-fem."""

        @BilinearForm
        def a(u, v, w):
            terme_masse      = (self.rho * self.cp / dt) * u * v
            terme_conduction = (self.kx * u.grad[0] * v.grad[0]
                              + self.ky * u.grad[1] * v.grad[1])
            return terme_masse + terme_conduction

        @LinearForm
        def L(v, w):
            chaleur_precedente = (self.rho * self.cp / dt) * w.T_old * v
            source_induction   = w.Q_ind * v
            return chaleur_precedente + source_induction

        return a, L

    def generer_source_induction(self, amplitude, rayon=0.015):
        """Modélise la bobine comme une tache gaussienne centrée en (0,0)."""
        x = self.basis.doflocs[0]
        y = self.basis.doflocs[1]
        return amplitude * np.exp(-(x**2 + y**2) / (2 * rayon**2))

    # ------------------------------------------------------------------
    # ★ MÉTHODE MODIFIÉE : simuler()
    # ------------------------------------------------------------------

    def simuler(self, amplitude_q, temps_cible=10.0, dt=1.0, T_init=22.0):
        """
        Intègre l'équation de la chaleur de t=0 jusqu'à t=temps_cible.

        NOUVEAUTÉ : à chaque pas de temps, on enregistre :
          - le temps courant t
          - les températures virtuelles aux 4 TC

        Retour
        ------
        historique : dict avec les clés suivantes
            'temps'  : list[float]          — instants enregistrés
            'TC2'    : list[float]           — température TC2 à chaque instant
            'TC3'    : list[float]
            'TC4'    : list[float]
            'TC5'    : list[float]
            'T_final': np.ndarray            — champ nodal complet au dernier pas

        Pourquoi un dictionnaire ?
        --------------------------
        Un dict est plus lisible et plus robuste qu'un tuple de 5 listes.
        On peut ajouter un 5e TC sans toucher à toutes les lignes de code
        qui consomment ce résultat (visualisation, calibration…).
        """
        a, L = self.formuler_probleme(dt)
        A    = asm(a, self.basis)                       # Matrice de rigidité (constante)

        T_actuelle = np.full(self.basis.N, T_init)      # Condition initiale uniforme
        Q_ind      = self.generer_source_induction(amplitude_q)

        # ── Initialisation de l'historique ────────────────────────────
        # On crée un dict dont les valeurs sont des LISTES Python.
        # Les listes permettent l'ajout en O(1) avec .append() à chaque pas.
        historique = {nom: [] for nom in self.coords_tc}   # TC2, TC3, TC4, TC5
        historique['temps']   = []
        historique['T_final'] = None

        # Enregistrement de l'état initial (t = 0)
        historique['temps'].append(0.0)
        for nom, temp in self._extraire_temperatures_tc(T_actuelle).items():
            historique[nom].append(temp)

        # ── Boucle de temps ───────────────────────────────────────────
        for t in np.arange(dt, temps_cible + dt, dt):
            # Assemblage du second membre avec les données du pas précédent
            b          = asm(L, self.basis, T_old=T_actuelle, Q_ind=Q_ind)
            T_actuelle = solve(A, b)

            # Enregistrement du temps courant (arrondi pour éviter les
            # erreurs flottantes comme 9.999999... au lieu de 10.0)
            t_enregistre = round(float(t), 10)
            historique['temps'].append(t_enregistre)

            # Extraction et stockage des températures aux 4 TC
            for nom, temp in self._extraire_temperatures_tc(T_actuelle).items():
                historique[nom].append(temp)

        historique['T_final'] = T_actuelle
        return historique

    # ------------------------------------------------------------------
    # Calibration (mise à jour pour utiliser le nouvel historique)
    # ------------------------------------------------------------------

    def calibrer_modele(self, vraies_temperatures_a_t_cible, temps_cible=10.0, dt=1.0):
        """
        Trouve l'amplitude Q qui minimise l'erreur quadratique entre la
        simulation et les mesures réelles aux 4 TC au temps temps_cible.

        Paramètres
        ----------
        vraies_temperatures_a_t_cible : dict
            Ex. {'TC2': 250.0, 'TC3': 40.0, 'TC4': 30.0, 'TC5': 30.0}
        """

        def fonction_erreur(amplitude_test):
            amp = float(amplitude_test[0])

            # simuler() retourne maintenant l'historique complet ;
            # on n'a besoin que du DERNIER instant → T_final
            hist = self.simuler(amplitude_q=amp,
                                temps_cible=temps_cible,
                                dt=dt)

            # Températures simulées au dernier pas de temps
            T_simul_tc = self._extraire_temperatures_tc(hist['T_final'])

            erreur_totale = sum(
                (T_simul_tc[nom] - vraies_temperatures_a_t_cible[nom])**2
                for nom in self.coords_tc
            )
            return erreur_totale

        print("Démarrage du calibrage du jumeau numérique...")
        resultat = minimize(fonction_erreur, x0=[1e6], method='Nelder-Mead',
                            options={'xatol': 1e3, 'fatol': 1.0, 'disp': True})
        amp_opt = float(resultat.x[0])
        print(f"Calibrage terminé ! Amplitude optimale : {amp_opt:.2e} W/m³")
        return amp_opt


# ─────────────────────────────────────────────────────────────────────
# Exemple d'utilisation
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    jumeau = JumeauNumeriqueInduction()

    temperatures_reelles_t10 = {
        'TC2': 250.0,
        'TC3':  40.0,
        'TC4':  30.0,
        'TC5':  30.0,
    }

    # 1. Calibration
    puissance_optimale = jumeau.calibrer_modele(temperatures_reelles_t10)

    # 2. Simulation complète avec historique
    historique = jumeau.simuler(amplitude_q=puissance_optimale, temps_cible=10.0)

    # 3. Affichage rapide de l'historique
    print("\nTemps enregistrés :", historique['temps'])
    print("Températures TC2  :", [f"{t:.1f}" for t in historique['TC2']])