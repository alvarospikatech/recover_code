import pyvista as pv
import numpy as np
import trimesh
import trimesh.repair

esfera = pv.Sphere()
bola_caras = esfera.faces.reshape(esfera.n_faces, 4)[:, 1:]
print("3xF caras",bola_caras)
bola_ps = esfera.points

b0 = bola_caras[0][0]
b1 = bola_caras[0][1]
b2 = bola_caras[0][2]

print(bola_caras)
print(np.insert(bola_caras[1:], 0, np.array([b1, b2, b0])))
nuevas_caras = np.insert(bola_caras[1:], 0, np.array([b1, b0, b2]), axis = 0)
print(len(nuevas_caras))
bola = trimesh.Trimesh(bola_ps, nuevas_caras)



print("winding?" , bola.is_winding_consistent)
trimesh.repair.fix_winding(bola)
print("repaired?" , bola.is_winding_consistent)

