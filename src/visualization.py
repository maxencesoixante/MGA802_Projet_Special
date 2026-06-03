import matplotlib.pyplot as plt
from skfem.visuals.matplotlib import plot


def afficher_champ_temperature(mesh, T_field, titre="Distribution de température à l'interface"):
    """
    Affiche la carte de chaleur 2D de la température calculée par scikit-fem.
    """
    fig, ax = plt.subplots(figsize=(6, 8))
    # plot() trace la solution sur le maillage
    plot(mesh, T_field, ax=ax, shading='gouraud', colorbar=True)

    # Ajout des marqueurs pour visualiser où étaient les thermocouples
    ax.plot(0, 0, 'wo', label="TC2")
    ax.plot(0, 0.060, 'ro', label="TC3")
    ax.plot(0.020, -0.060, 'go', label="TC4")
    ax.plot(-0.020, -0.060, 'bo', label="TC5")

    ax.set_title(titre)
    ax.set_xlabel("Largeur X (m)")
    ax.set_ylabel("Longueur Y (m)")
    ax.legend()
    plt.tight_layout()
    plt.show()