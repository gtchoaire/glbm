import numpy as np
from matplotlib import pyplot as plt

path = "../build/output/20261014_1630"

time = np.loadtxt(f"{path}/total_kinetic_energy.dat", unpack=True, usecols=(0))
tke = np.loadtxt(f"{path}/total_kinetic_energy.dat", unpack=True, usecols=(1))

plt.plot(time, tke)
plt.xlabel("Time")
plt.ylabel("Total Kinetic Energy")
plt.show()
