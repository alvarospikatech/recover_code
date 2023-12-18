import scipy.io as sci
import numpy as np
import trimesh
import pandas as pd
from scipy.interpolate import RBFInterpolator

def rbf_interpolation(
    nodes: np.ndarray,
    measured: np.ndarray,
    to_interpolate: np.ndarray,
    kernel: str = "thin_plate_spline",
    epsilon: float = 1,
) -> np.ndarray:
    """
    Interpolates the measured potentials using RBF interpolation.
    Args:
        nodes (np.ndarray): The nodes of the mesh
        measured (np.ndarray): The measured potentials
        to_interpolate (np.ndarray): The channels that were not measured correctly
        kernel (str): The kernel to use
        epsilon (float): The epsilon value
    Returns:
        np.ndarray: The interpolated potentials
    """
    to_interpolate_mask = np.zeros(shape=len(nodes), dtype=bool)
    to_interpolate_mask[to_interpolate] = True

    known_nodes = nodes[~to_interpolate_mask, :]
    known_measures = measured.T
    unknown_nodes = nodes[to_interpolate_mask, :]
    interpolator = RBFInterpolator(known_nodes, known_measures, kernel=kernel, epsilon=epsilon)
    est_potentials = interpolator(unknown_nodes)

    interpolated = np.zeros((nodes.shape[0],known_measures.shape[1]))
    interpolated[to_interpolate_mask, :] = est_potentials
    interpolated[~to_interpolate_mask, :] = known_measures
    
    
    return interpolated

def _interpolate_signals(points, signals, bad_electrodes, electrode_indexes):
    idx_to_interpolate = set(np.arange(points.shape[0]))
    idx_to_interpolate = idx_to_interpolate - (
        set(electrode_indexes)
    )
    idx_to_interpolate = list(idx_to_interpolate)
    interpola = rbf_interpolation(points, signals.T, idx_to_interpolate)
    return interpola

def loadDataSet(value):
    
    #valencia_sintetic = "Data"


    if value == "valencia_sintetic_la_fibrotic":

        heartPotentials = sci.loadmat("valencia_sintetic" + '/Interventions/fibrotic_tissue/EGM_LA_fibrotic.mat')['EGM']['potvals'][0][0]
        heartMesh_nodes = sci.loadmat("valencia_sintetic/atria" + '/atria.mat')['Atria']['node'][0][0]
        heartMesh_faces = sci.loadmat("valencia_sintetic/atria" + '/atria.mat')['Atria']['face'][0][0]
        fs = 500

        return np.array(heartPotentials) , np.array(heartMesh_nodes), np.array(heartMesh_faces - 1),fs
    

    if value == "valencia_sintetic_la_normal":

        heartPotentials = sci.loadmat("valencia_sintetic" + '/EGM_LA_normal/EGM_LA_normal.mat')['EGM']['potvals'][0][0]
        heartMesh_nodes = sci.loadmat("valencia_sintetic/atria" + '/atria.mat')['Atria']['node'][0][0]
        heartMesh_faces = sci.loadmat("valencia_sintetic/atria" + '/atria.mat')['Atria']['face'][0][0]
        fs = 500

        return np.array(heartPotentials) , np.array(heartMesh_nodes), np.array(heartMesh_faces - 1),fs
    
    if value == "valencia_sintetic_ra_normal":

        heartPotentials = sci.loadmat("valencia_sintetic"+ '/EGM_RA_normal/EGM_RA_normal.mat')['EGM']['potvals'][0][0]
        heartMesh_nodes = sci.loadmat("valencia_sintetic/atria" + '/atria.mat')['Atria']['node'][0][0]
        heartMesh_faces = sci.loadmat("valencia_sintetic/atria" + '/atria.mat')['Atria']['face'][0][0]
        fs = 500
        return np.array(heartPotentials) , np.array(heartMesh_nodes), np.array(heartMesh_faces - 1), fs
    

    if value == "valencia_real_pat1_old":

        heartPotentials_leads = sci.loadmat("valencia_pat1"+ '/Interventions/AV_block/EGM_AV_block.mat')['EGM']['potvals'][0][0]
        mesh_index = sci.loadmat("valencia_pat1"+ '/Interventions/AV_block/EGM_AV_block.mat')['EGM']['mesh_index'][0][0]

        heartPotentials_leads = np.nan_to_num(heartPotentials_leads)
        heartMesh_nodes = sci.loadmat("valencia_pat1/Meshes" + '/atria.mat')['Atria']['node'][0][0]
        heartMesh_faces = sci.loadmat("valencia_pat1/Meshes" + '/atria.mat')['Atria']['face'][0][0]
        fs = 2.0345e3


        heartPotentials = np.zeros((heartMesh_nodes.shape[0],heartPotentials_leads.shape[1]))
        for i in np.arange(0,mesh_index.shape[0]):
            heartPotentials[mesh_index[i]] = heartPotentials_leads[i]
        

        heartPotentials = np.nan_to_num(heartPotentials)
        return np.array(heartPotentials) , np.array(heartMesh_nodes), np.array(heartMesh_faces - 1), fs
    
    if value == "valencia_real_pat1":

        heartPotentials_leads = sci.loadmat("valencia_pat1"+ '/Interventions/AV_block/EGM_AV_block.mat')['EGM']['potvals'][0][0]
        mesh_index = sci.loadmat("valencia_pat1"+ '/Interventions/AV_block/EGM_AV_block.mat')['EGM']['mesh_index'][0][0]
        heartPotentials_leads = np.nan_to_num(heartPotentials_leads)
        heartMesh_nodes = sci.loadmat("valencia_pat1/Meshes" + '/atria.mat')['Atria']['node'][0][0]
        heartMesh_faces = sci.loadmat("valencia_pat1/Meshes" + '/atria.mat')['Atria']['face'][0][0]
        fs = 2.0345e3
        mesh_index = mesh_index.flatten().tolist()
        
        heartPotentials = np.zeros((heartMesh_nodes.shape[0],heartPotentials_leads.shape[1]))
        for i in np.arange(0,len(mesh_index)):
            heartPotentials[mesh_index[i]] = heartPotentials_leads[i]
        heartPotentials = _interpolate_signals(heartMesh_nodes, heartPotentials_leads, [], mesh_index)
        #heartPotentials = heartPotentials[:,8000:]
        return np.array(heartPotentials) , np.array(heartMesh_nodes), np.array(heartMesh_faces - 1), fs
    
    if value == "valencia_real_pat2":

        heartPotentials_leads = sci.loadmat("valencia_pat2"+ '/Interventions/AV_block/EGM_AV_block.mat')['EGM']['potvals'][0][0]
        mesh_index = sci.loadmat("valencia_pat2"+ '/Interventions/AV_block/EGM_AV_block.mat')['EGM']['mesh_index'][0][0]
        heartPotentials_leads = np.nan_to_num(heartPotentials_leads)
        heartMesh_nodes = sci.loadmat("valencia_pat2/Meshes" + '/atria.mat')['Atria']['node'][0][0]
        heartMesh_faces = sci.loadmat("valencia_pat2/Meshes" + '/atria.mat')['Atria']['face'][0][0]
        fs = 2.0345e3
        mesh_index = mesh_index.flatten().tolist()

        
        heartPotentials = np.zeros((heartMesh_nodes.shape[0],heartPotentials_leads.shape[1]))
        for i in np.arange(0,len(mesh_index)):
            heartPotentials[mesh_index[i]] = heartPotentials_leads[i]
        heartPotentials = _interpolate_signals(heartMesh_nodes, heartPotentials_leads, [], mesh_index)
        return np.array(heartPotentials) , np.array(heartMesh_nodes), np.array(heartMesh_faces - 1), fs
    

    if value.startswith("chaleco_2"):
        chalec_dataset = value[10:]
        heart_load  = trimesh.load_mesh("datos_chaleco_2/" + chalec_dataset + "/heart_pos.stl")
        heartMesh_nodes= heart_load.vertices * 1000
        heartMesh_faces = heart_load.faces
        data_frame = pd.read_csv("datos_chaleco_2/" + chalec_dataset + "/heart_potentials_2.csv")
        heartPotentials = data_frame.values.T
        fs = 360
        return np.array(heartPotentials) , np.array(heartMesh_nodes), np.array(heartMesh_faces), fs
    

    elif value.startswith("chaleco_1"):
        chalec_dataset = value[10:]
        heart_load  = trimesh.load_mesh("datos_chaleco/" + chalec_dataset + "/heart_pos.stl")
        heartMesh_nodes= heart_load.vertices * 1000
        heartMesh_faces = heart_load.faces
        data_frame = pd.read_csv("datos_chaleco/" + chalec_dataset + "/heart_potentials_2.csv")
        heartPotentials = data_frame.values.T
        fs = 360
        return np.array(heartPotentials) , np.array(heartMesh_nodes), np.array(heartMesh_faces), fs
    

    elif value.startswith("pruebas_alex"):
        chalec_dataset = value[13:]
        heart_load  = trimesh.load_mesh("pruebas_alex/" + chalec_dataset + "/heart_pos.stl")
        heartMesh_nodes= heart_load.vertices * 1000
        heartMesh_faces = heart_load.faces
        data_frame = pd.read_csv("pruebas_alex/" + chalec_dataset + "/heart_potentials.csv")
        heartPotentials = data_frame.values.T
        fs = 360
        return np.array(heartPotentials) , np.array(heartMesh_nodes), np.array(heartMesh_faces), fs