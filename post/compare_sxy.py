
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

output_dir = Path("../build/output/20261113_0731")

NX = 128
NZ = 128
STEP = 992188

for x in [0,1,2, NX // 2, NX - 3, NX - 2, NX - 1]:

    filename = output_dir / (
        f"sxy_x{x}_z{NZ//2}_step{STEP:08d}.csv"
    )

    data = np.genfromtxt(
        filename,
        delimiter=",",
        names=True
    )
    sxy_fd = 0.5 * (data["dux_dy"] + data["duy_dx"])
    sxy_mrlbm = data["sxy_moment"]
    sxy_original = data["sxy_moment"]
    sxy_explicit = data["sxy_explicit"]
    sxy_fd = data["sxy_fd"]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(sxy_original, data["y_star"],
        "k-", label="MR-LBM original")

    ax.plot(sxy_explicit, data["y_star"],
            "b--", label="MR-LBM sem escalonamento")

    ax.plot(sxy_fd, data["y_star"],
            "r:", label="Diferenças finitas")

    print(
    "Diferenca maxima entre formulas:",
    np.max(np.abs(sxy_original - sxy_explicit))
    )

    # ax.plot(
    #     sxy_fd,
    #     data["y_star"],
    #     "r--",
    #     label=r"$S_{xy}$ - Diferenças finitas"
    # )

    # ax.plot(
    #     sxy_mrlbm,
    #     data["y_star"],
    #     "k-",
    #     label=r"$S_{xy}$ - MR-LBM"
    # )

    # ax.plot(
    #     0.5 * data["dux_dy"],
    #     data["y_star"],
    #     "b-",
    #     label=r"$\frac{1}{2}\partial_y u_x$"
    # )

    # ax.plot(
    #     0.5 * data["duy_dx"],
    #     data["y_star"],
    #     "g--",
    #     label=r"$\frac{1}{2}\partial_x u_y$"
    # )

    # ax.plot(
    #     data["sxy_moment"],
    #     data["y_star"],
    #     "k:",
    #     label=r"$S_{xy}$ MR-LBM"
    # )

    ax.set_xlabel(r"$S_{xy}$")
    ax.set_ylabel(r"$y/L$")
    ax.set_xlim(-0.05,0.05)
    ax.legend()
    ax.grid(alpha=0.3)
    # y = data["y_star"]
    # error = sxy_mrlbm - sxy_fd

    # regions = {
    #     "Interior": (y >= 0.05) & (y <= 0.80),
    #     "Proximo a tampa": (y > 0.80) & (y < 1.0)
    # }

    # for name, mask in regions.items():

    #     rmse = np.sqrt(np.mean(error[mask] ** 2))
    #     linf = np.max(np.abs(error[mask]))

    #     print(
    #         f"x = {x} | {name} | "
    #         f"RMSE = {rmse:.6e} | "
    #         f"Linf = {linf:.6e}"
    #     )

plt.show()


