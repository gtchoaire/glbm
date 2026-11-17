import numpy as np
from pathlib import Path

output_dir = Path("../build/output/20261112_0942")

NX = NY = NZ = 64
N_STEPS = 10

x = 3 * NX // 4
y = NY - 2
z = NZ // 2

index = x + NX * (y + NY * z)

for field in ["rho", "ux", "uy", "uz"]:

    values = []

    for step in range(1, N_STEPS + 1):
        filename = output_dir / f"{field}_{step:08d}.bin"

        data = np.fromfile(filename, dtype=np.float32)

        values.append(data[index])

    average = np.mean(values, dtype=np.float64)

    print(f"{field}_avg = {average:.10e}")