import numpy as np

personality1 = np.array([0.1, 0.2, 0.3, 0.45, 0.5])
personality2 = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
personality3 = np.array([0.10, 0.90, 0.10, 0.60, 0.40])


def dot_product(vec1, vec2):
    return np.dot(vec1, vec2)

print(f" Dot Product of per1 * per2: {dot_product(personality1, personality2):.02f}")
print(f" Dot Product of per1 * per3: {dot_product(personality1, personality3):.02f}")
print(f" Dot Product of per2 * per3: {dot_product(personality2, personality3):.02f}")



