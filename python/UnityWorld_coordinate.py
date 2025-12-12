import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

# PIPELINE (UNITY MESH → RB LOCAL → MOTIVE WORLD)

# 1) UNITY MESH POINTS → correct with conversion matrix C
C = np.array([
    [-1, 0,  0],
    [ 0, 1,  0],
    [ 0, 0, -1],
], dtype=float)

# Tibia mesh points 
meshtibia_local_unity = np.array([
    [-0.02015125,  0.4032193,   0.03225459],  # AMT
    [ 0.03259825,  0.01721638, -0.01066572],  # LT
    [-0.03612585,  0.02022061,  0.01776088],  # MT
], dtype=float).T

# Femur mesh points 
meshfemur_local_unity = np.array([
    [-0.002581612, -0.4347238,   0.04451473],   # TF=AF
    [ 0.03755924,  -0.001844588, -0.01695582],  # LF
    [-0.03962225,  -0.004840371,  0.01212757],  # MF
], dtype=float).T

# Convert Unity → Blender-like (used for Kabsch)
meshtibia_local = C @ meshtibia_local_unity
meshfemur_local = C @ meshfemur_local_unity

# 2) RB-LOCAL marker positions (from Motive VP)
tibia_vp_table = pd.read_csv("../data/csv_files/tibiaVP.csv")
femur_vp_table = pd.read_csv("../data/csv_files/femurVP.csv")

tibia_points_rb = np.array([
    [tibia_vp_table["tibiaAMTlocal_X"].iloc[0], tibia_vp_table["tibiaAMTlocal_Y"].iloc[0], tibia_vp_table["tibiaAMTlocal_Z"].iloc[0]],
    [tibia_vp_table["tibiaLTlocal_X"].iloc[0],  tibia_vp_table["tibiaLTlocal_Y"].iloc[0],  tibia_vp_table["tibiaLTlocal_Z"].iloc[0]],
    [tibia_vp_table["tibiaMTlocal_X"].iloc[0],  tibia_vp_table["tibiaMTlocal_Y"].iloc[0],  tibia_vp_table["tibiaMTlocal_Z"].iloc[0]],
], dtype=float).T / 1000.0  # mm → m

femur_points_rb = np.array([
    [femur_vp_table["femurAFlocal_X"].iloc[0], femur_vp_table["femurAFlocal_Y"].iloc[0], femur_vp_table["femurAFlocal_Z"].iloc[0]],
    [femur_vp_table["femurLFlocal_X"].iloc[0], femur_vp_table["femurLFlocal_Y"].iloc[0], femur_vp_table["femurLFlocal_Z"].iloc[0]],
    [femur_vp_table["femurMFlocal_X"].iloc[0], femur_vp_table["femurMFlocal_Y"].iloc[0], femur_vp_table["femurMFlocal_Z"].iloc[0]],
], dtype=float).T / 1000.0

# KABSCH helper (A,B are 3xN)
def kabsch_rt(A: np.ndarray, B: np.ndarray):
    centroid_a = np.mean(A, axis=1, keepdims=True)
    centroid_b = np.mean(B, axis=1, keepdims=True)

    A0 = A - centroid_a
    B0 = B - centroid_b

    H = A0 @ B0.T
    U, _, Vt = np.linalg.svd(H)

    V = Vt.T
    R_ab = V @ U.T

    if np.linalg.det(R_ab) < 0:
        V[:, -1] *= -1
        R_ab = V @ U.T

    T_ab = centroid_b - R_ab @ centroid_a
    return R_ab, T_ab

# 3) KABSCH: compute mesh → RB transform 
R_mesh_rb_tibia, T_mesh_rb_tibia = kabsch_rt(meshtibia_local, tibia_points_rb)
R_mesh_rb_femur, T_mesh_rb_femur = kabsch_rt(meshfemur_local, femur_points_rb)

# 4) Load RB → WORLD transforms (Motive)
motive_tibia = pd.read_csv("../data/csv_files/tibiaTransform.csv")
motive_femur = pd.read_csv("../data/csv_files/femurTransform.csv")

n_frames = len(motive_tibia)

rb_world_tibia_wxyz = np.column_stack([
    motive_tibia["Tibia_Rotation_W"].to_numpy(),
    motive_tibia["Tibia_Rotation_X"].to_numpy(),
    motive_tibia["Tibia_Rotation_Y"].to_numpy(),
    motive_tibia["Tibia_Rotation_Z"].to_numpy(),
])

rb_world_femur_wxyz = np.column_stack([
    motive_femur["Femur_Rotation_W"].to_numpy(),
    motive_femur["Femur_Rotation_X"].to_numpy(),
    motive_femur["Femur_Rotation_Y"].to_numpy(),
    motive_femur["Femur_Rotation_Z"].to_numpy(),
])

# Translation (mm → m)
translation_tibia = np.vstack([
    motive_tibia["Tibia_Position_X"].to_numpy(),
    motive_tibia["Tibia_Position_Y"].to_numpy(),
    motive_tibia["Tibia_Position_Z"].to_numpy(),
]) / 1000.0

translation_femur = np.vstack([
    motive_femur["Femur_Position_X"].to_numpy(),
    motive_femur["Femur_Position_Y"].to_numpy(),
    motive_femur["Femur_Position_Z"].to_numpy(),
]) / 1000.0

# 5) meshUnity → RB → Motive world
mesh_world_tibia = np.zeros((n_frames, 8))
mesh_world_femur = np.zeros((n_frames, 8))

def rotm_to_matlab_wxyz(rotm: np.ndarray):
    # scipy returns [x, y, z, w]; MATLAB wants [w, x, y, z]
    q_xyzw = R.from_matrix(rotm).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)

def motive_wxyz_to_rotm(q_wxyz: np.ndarray):
    # scipy expects [x, y, z, w]
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=float)
    return R.from_quat(q_xyzw).as_matrix()

for i in range(n_frames):

    # ---------- TIBIA ----------
    R_rb_world = motive_wxyz_to_rotm(rb_world_tibia_wxyz[i])
    T_rb_world = translation_tibia[:, i].reshape(3, 1)

    R_world = R_rb_world @ R_mesh_rb_tibia
    T_world = R_rb_world @ T_mesh_rb_tibia + T_rb_world

    q_wxyz = rotm_to_matlab_wxyz(R_world)  # [qw qx qy qz]
    mesh_world_tibia[i, :] = np.array([i, q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0], T_world[0,0], T_world[1,0], T_world[2,0]])

    # ---------- FEMUR ----------
    R_rb_world = motive_wxyz_to_rotm(rb_world_femur_wxyz[i])
    T_rb_world = translation_femur[:, i].reshape(3, 1)

    R_world = R_rb_world @ R_mesh_rb_femur
    T_world = R_rb_world @ T_mesh_rb_femur + T_rb_world

    q_wxyz = rotm_to_matlab_wxyz(R_world)
    mesh_world_femur[i, :] = np.array([i, q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0], T_world[0,0], T_world[1,0], T_world[2,0]])

headers = ["Frame", "qx", "qy", "qz", "qw", "tx", "ty", "tz"]

pd.DataFrame(mesh_world_tibia, columns=headers).to_csv("../data/csv_files/tibiaTransformForUnity.csv", index=False)
pd.DataFrame(mesh_world_femur, columns=headers).to_csv("../data/csv_files/femurTransformForUnity.csv", index=False)

print("CSV for Unity written.")