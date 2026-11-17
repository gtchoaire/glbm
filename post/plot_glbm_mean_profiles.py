#!/usr/bin/env python3
"""Plot time-averaged Sxy profiles extracted from a GLBM checkpoint v3.

Usage:
    python3 plot_glbm_mean_profiles.py post/mean_profiles
"""
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, help='Folder with sxy_mean_x*.csv')
    args = parser.parse_args()

    csv_files = sorted(args.folder.glob('sxy_mean_x*.csv'))
    if not csv_files:
        raise SystemExit(f'No mean-profile CSV files found in {args.folder}')

    for csv_file in csv_files:
        data = np.genfromtxt(csv_file, delimiter=',', names=True)
        x = int(data['x'][0])
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
        for ax in axes:
            ax.plot(data['sxy_avg'], data['y_star'], 'k-', lw=1.5,
                    label=r'$\overline{S_{xy}}$ (MR-LBM)')
            ax.plot(data['sxy_fd'], data['y_star'], 'r--', lw=1.5,
                    label=r'$\frac{1}{2}(\partial_y\overline{u_x}+\partial_x\overline{u_y})$')
            ax.axvline(0, color='0.7', lw=0.7)
            ax.set_ylim(0, 1)
            ax.set_xlabel(r'$S_{xy}$ (lattice units)')
            ax.grid(alpha=0.25)
        axes[0].set_xlim(-0.05, 0.05)  # same horizontal scale as old budget plots
        axes[0].set_title('Escala original: [-0.05, 0.05]')
        axes[1].set_title('Zoom automático')
        axes[0].set_ylabel(r'$y/L$')
        axes[1].legend(loc='best', fontsize=8)
        fig.suptitle(f'Campos médios - x = {x}')
        fig.tight_layout()
        target = csv_file.with_suffix('.png')
        fig.savefig(target, dpi=170)
        plt.close(fig)
        print(target)


if __name__ == '__main__':
    main()
