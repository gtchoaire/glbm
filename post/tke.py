import numpy as np
from matplotlib import pyplot as plt

path = "../build/output/20261015_1023"

time = np.loadtxt(f"{path}/total_kinetic_energy.dat", unpack=True, usecols=(0))
tke = np.loadtxt(f"{path}/total_kinetic_energy.dat", unpack=True, usecols=(1))

tstar = (time*0.0256)/255

plt.plot(tstar, tke)
plt.xlabel(r"$t^*$")
plt.ylabel("Total Kinetic Energy")
plt.show()
