import numpy as np
import pandas as pd
from skfem import *
from scipy.optimize import minimize


class JumeauNumeriqueInduction:
    def __init__(self, config: dict):
        """
        Initialise le jumeau numérique depuis un dictionnaire de configuration.

        Pourquoi passer un dict plutôt que des arguments individuels ?
        --------------------------------------------------------------
        Avec 15+ paramètres physiques, une signature `__init__(rho, cp, kx, …)`
        devient illisible et fragile (ordre des arguments). Un dict nommé isole
        complètement la définition des paramètres (config.yaml) de leur usage
        (ce constructeur). C'est le patron de conception "Dependency Injection" :
        la classe ne sait pas d'où vient la config — fichier, test unitaire,
        interface graphique — ce qui la rend testable et réutilisable.

        Paramètres
        ----------
        config : dict
            Dictionnaire chargé depuis config.yaml (via yaml.safe_load).
            Sections attendues : materiau, geometrie, maillage,
                                 thermocouples, source_induction,
                                 simulation, calibration.
        """
        # ── 1. Lecture des sections de configuration ──────────────────
        mat  = config['materiau']
        geo  = config['geometrie']
        mail = config['maillage']
        src  = config['source_induction']
        sim  = config['simulation']
        cal  = config['calibration']

        # ── 2. Propriétés thermophysiques ─────────────────────────────
        self.rho = float(mat['rho'])   # Densité [kg/m³]
        self.cp  = float(mat['cp'])    # Capacité thermique [J/kg·K]
        self.kx  = float(mat['kx'])    # Conductivité transverse [W/m·K]
        self.ky  = float(mat['ky'])    # Conductivité longitudinale [W/m·K]

        # ── 3. Paramètres de simulation ───────────────────────────────
        self.dt          = float(sim['dt'])
        self.T_initiale  = float(sim['T_initiale'])
        self.T_ambiante  = float(sim['T_ambiante'])
        self.rayon_bobine = float(src['rayon_bobine'])

        # ── 4. Paramètres de calibration (stockés pour calibrer_modele) ─
        self._cal = cal   # conservé complet pour y accéder dans calibrer_modele()

        # ── 5. Création du maillage 2D ────────────────────────────────
        self.mesh = MeshTri.init_tensor(
            np.linspace(float(geo['x_min']), float(geo['x_max']), int(mail['nx'])),
            np.linspace(float(geo['y_min']), float(geo['y_max']), int(mail['ny']))
        )
        self.basis = Basis(self.mesh, ElementTriP1())

        # ── 6. Coordonnées des thermocouples ──────────────────────────
        # Le YAML stocke chaque TC comme une liste [x, y].
        # On convertit en dict de tuples pour le reste du code.
        tc_raw = config['thermocouples']
        self.coords_tc = {nom: tuple(coords) for nom, coords in tc_raw.items()}

        # ── 7. Pré-calcul des indices nodaux les plus proches ─────────
        # Fait une seule fois ici — O(N_noeuds) par TC, amorti sur toute
        # la simulation (économise 4 × N_pas recherches).
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

    def formuler_probleme(self, dt, h_eff):
        """
        Forme faible de Galerkin pour l'équation de la chaleur 2D avec
        terme de Robin volumique (pertes vers l'outillage + ambiance).

        Forme bilinéaire  a(u,v) = ∫ [(ρcp/Δt + h_eff)·u·v + k∇u·∇v] dΩ
        Forme linéaire    L(v)   = ∫ [(ρcp/Δt·T^n + Q_ind + h_eff·T_amb)·v] dΩ

        h_eff est reçu en paramètre (calibré par calibrer_modele).
        T_ambiante provient de self (config.yaml → section simulation).
        """
        T_amb = self.T_ambiante

        @BilinearForm
        def a(u, v, w):
            terme_masse      = (self.rho * self.cp / dt) * u * v
            terme_conduction = (self.kx * u.grad[0] * v.grad[0]
                              + self.ky * u.grad[1] * v.grad[1])
            terme_pertes     = h_eff * u * v   # implicite → stabilité garantie
            return terme_masse + terme_conduction + terme_pertes

        @LinearForm
        def L(v, w):
            chaleur_precedente = (self.rho * self.cp / dt) * w.T_old * v
            source_induction   = w.Q_ind * v
            rappel_ambiant     = h_eff * T_amb * v
            return chaleur_precedente + source_induction + rappel_ambiant

        return a, L

    def generer_source_induction(self, amplitude):
        """
        Champ source gaussien Q(x,y) = amplitude · exp(-(x²+y²)/(2σ²)).
        σ = self.rayon_bobine, lu depuis config.yaml (section source_induction).
        """
        x = self.basis.doflocs[0]
        y = self.basis.doflocs[1]
        sigma = self.rayon_bobine
        return amplitude * np.exp(-(x**2 + y**2) / (2 * sigma**2))

    # ------------------------------------------------------------------
    # ★ MÉTHODE MODIFIÉE : simuler()
    # ------------------------------------------------------------------

    def simuler(self, amplitude_q, h_eff=5000.0, temps_cible=None,
                dt=None, T_init=None, temps_arret=None):
        """
        Intègre l'équation de la chaleur de t=0 jusqu'à t=temps_cible.

        Nouveauté : sauvegarde optimisée des champs 2D complets
        --------------------------------------------------------
        À chaque pas, le champ nodal T (vecteur de longueur N_noeuds) est
        copié dans une matrice NumPy pré-allouée de forme (N_pas+1, N_noeuds).

        Analyse mémoire :
          N_noeuds ≈ 1200  |  N_pas ≈ 300  |  float64 = 8 octets
          Mémoire ≈ 1200 × 300 × 8 = 2.9 Mo  → parfaitement gérable en RAM.

        La pré-allocation (np.empty) est préférable aux .append() successifs
        car elle évite les ré-allocations dynamiques et permet un accès O(1)
        à n'importe quelle tranche temporelle : champs_2d[i] = T au pas i.

        Paramètres
        ----------
        amplitude_q  : float — amplitude de la source gaussienne [W/m³]
        h_eff        : float — coefficient de perte volumique [W/m³·K]
        temps_cible  : float — durée totale [s] (défaut : config simulation)
        dt           : float — pas de temps [s] (défaut : config simulation)
        T_init       : float — température initiale [°C] (défaut : config)
        temps_arret  : float | None — instant d'extinction de la bobine [s]

        Retour — historique : dict
        --------------------------
        'temps'     : np.ndarray (N_pas+1,)          instants [s]
        'TC2'…'TC5' : np.ndarray (N_pas+1,)          températures aux TC [°C]
        'champs_2d' : np.ndarray (N_pas+1, N_noeuds) champs nodaux complets [°C]
        'T_final'   : np.ndarray (N_noeuds,)          alias sur champs_2d[-1]
        """
        # Valeurs par défaut depuis la config (si non passées en argument)
        if dt       is None: dt      = self.dt
        if T_init   is None: T_init  = self.T_initiale
        if temps_cible is None:
            raise ValueError("temps_cible doit être fourni à simuler().")

        a, L = self.formuler_probleme(dt, h_eff)
        A    = asm(a, self.basis)

        # Pré-allocation de la matrice des champs 2D
        pas_temps    = np.arange(dt, temps_cible + dt, dt)
        N_pas        = len(pas_temps)
        N_noeuds     = self.basis.N
        champs_2d    = np.empty((N_pas + 1, N_noeuds), dtype=np.float64)

        T_actuelle = np.full(N_noeuds, T_init)
        champs_2d[0] = T_actuelle   # état initial stocké en ligne 0

        Q_ind_actif = self.generer_source_induction(amplitude_q)
        Q_ind_nul   = np.zeros(N_noeuds)

        # Tableaux pour l'historique 1D des TC (pré-alloués aussi)
        temps_arr = np.empty(N_pas + 1)
        tc_arr    = {nom: np.empty(N_pas + 1) for nom in self.coords_tc}

        # Enregistrement t = 0
        temps_arr[0] = 0.0
        for nom, temp in self._extraire_temperatures_tc(T_actuelle).items():
            tc_arr[nom][0] = temp

        # ── Boucle temporelle ─────────────────────────────────────────
        for i, t in enumerate(pas_temps):
            Q_courant = (Q_ind_nul
                         if (temps_arret is not None and t > temps_arret)
                         else Q_ind_actif)

            b          = asm(L, self.basis, T_old=T_actuelle, Q_ind=Q_courant)
            T_actuelle = solve(A, b)

            idx = i + 1
            champs_2d[idx]   = T_actuelle          # copie du vecteur nodal
            temps_arr[idx]   = round(float(t), 10)
            for nom, temp in self._extraire_temperatures_tc(T_actuelle).items():
                tc_arr[nom][idx] = temp

        historique = {
            'temps'    : temps_arr,
            'champs_2d': champs_2d,   # (N_pas+1, N_noeuds) — accès par indice
            'T_final'  : champs_2d[-1],
        }
        historique.update(tc_arr)   # ajoute TC2, TC3, TC4, TC5
        return historique

    # ------------------------------------------------------------------
    # Calibration (mise à jour pour utiliser le nouvel historique)
    # ------------------------------------------------------------------

    def calibrer_modele(self, historique_tc2_experimental,
                         temps_arret, temps_cible):
        """
        Calibration bi-paramétrique (Q_ind, h_eff) par minimisation de la
        MSE sur l'historique complet du thermocouple central TC2.

        Stratégie de changement de variable (log-space)
        ------------------------------------------------
        Q_ind ≈ 10⁶ W/m³  et  h_eff ≈ 10³ W/m³·K → 3 ordres de grandeur.
        On optimise θ = [ln(Q), ln(h)] ∈ ℝ² pour normaliser l'espace.
        Retour à l'espace physique : Q = exp(θ₀),  h = exp(θ₁).
        Garantit Q > 0 et h > 0 sans contrainte explicite.

        Les hyperparamètres de l'optimiseur (méthode, tolérances, point de
        départ) sont lus depuis self._cal (section calibration de config.yaml).
        """
        cal       = self._cal
        temps_exp = np.asarray(historique_tc2_experimental['temps'], dtype=float)
        T_exp_tc2 = np.asarray(historique_tc2_experimental['TC2'],   dtype=float)
        dt        = self.dt

        iteration = [0]

        def fonction_cout(theta):
            amp   = float(np.exp(theta[0]))
            h_eff = float(np.exp(theta[1]))

            hist = self.simuler(
                amplitude_q=amp, h_eff=h_eff,
                temps_cible=temps_cible, dt=dt,
                temps_arret=temps_arret,
            )
            temps_sim = hist['temps']
            T_sim_tc2 = hist['TC2']

            t_min  = max(temps_sim[0],  temps_exp[0])
            t_max  = min(temps_sim[-1], temps_exp[-1])
            masque = (temps_exp >= t_min) & (temps_exp <= t_max)
            if masque.sum() < 2:
                return 1e12

            T_interp = np.interp(temps_exp[masque], temps_sim, T_sim_tc2)
            mse = float(np.mean((T_interp - T_exp_tc2[masque])**2))

            iteration[0] += 1
            if iteration[0] % 5 == 0:
                print(f"    Iter {iteration[0]:4d} | Q={amp:.3e} W/m³ | "
                      f"h={h_eff:.1f} W/m³·K | MSE={mse:.3f} °C²")
            return mse

        # Point de départ depuis config.yaml (log-space)
        theta0 = np.array([
            float(cal['log_Q_init']) * np.log(10),   # ln(10^log_Q_init)
            float(cal['log_h_init']) * np.log(10),
        ])

        print("=" * 60)
        print("  CALIBRATION MULTI-PARAMÈTRES — MSE sur courbe TC2")
        print(f"  Méthode : {cal['methode']}  |  Espace : log(Q), log(h)")
        print("=" * 60)

        resultat = minimize(
            fonction_cout, x0=theta0,
            method=cal['methode'],
            options={
                'maxiter': int(cal['max_iterations']),
                'ftol'   : float(cal['tolerance_f']),
                'gtol'   : float(cal['tolerance_g']),
            }
        )

        amp_opt   = float(np.exp(resultat.x[0]))
        h_eff_opt = float(np.exp(resultat.x[1]))

        print(f"\n  ✓ {resultat.message}")
        print(f"  ✓ Q_ind  = {amp_opt:.4e} W/m³")
        print(f"  ✓ h_eff  = {h_eff_opt:.2f} W/m³·K")
        print(f"  ✓ RMSE   = {np.sqrt(resultat.fun):.3f} °C")
        print("=" * 60)
        return amp_opt, h_eff_opt


# ─────────────────────────────────────────────────────────────────────
# Exemple d'utilisation
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml, pathlib
    cfg = yaml.safe_load(pathlib.Path("config.yaml").read_text(encoding="utf-8"))
    jumeau = JumeauNumeriqueInduction(cfg)
    print("Jumeau initialisé depuis config.yaml")
    print(f"  Maillage : {cfg['maillage']['nx']}×{cfg['maillage']['ny']} nœuds")
    print(f"  ρ={jumeau.rho} kg/m³  |  kx={jumeau.kx}  ky={jumeau.ky} W/m·K")