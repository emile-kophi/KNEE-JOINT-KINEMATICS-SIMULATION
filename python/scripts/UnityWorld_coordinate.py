import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from kabasch_algorithm  import kabsch_rt
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
tibia_vp_table = pd.read_csv("../../data/csv_files/tibiaVP.csv")
femur_vp_table = pd.read_csv("../../data/csv_files/femurVP.csv")

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

# 3) KABSCH: compute mesh → RB transform 
R_mesh_rb_tibia, T_mesh_rb_tibia = kabsch_rt(meshtibia_local, tibia_points_rb)
R_mesh_rb_femur, T_mesh_rb_femur = kabsch_rt(meshfemur_local, femur_points_rb)
# =========================================================
# SAVE mesh → RB TRANSFORMS (STATIC GEOMETRY)
# =========================================================

BASE_DIR = Path(__file__).resolve().parent          # python/
OUT_DIR = BASE_DIR.parent / "data" / "RmatrixAndT"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Tibia ---
np.savez(
    OUT_DIR / "mesh_to_rb_tibia.npz",
    R=R_mesh_rb_tibia,
    T=T_mesh_rb_tibia
)

# --- Femur ---
np.savez(
    OUT_DIR / "mesh_to_rb_femur.npz",
    R=R_mesh_rb_femur,
    T=T_mesh_rb_femur
)

print("Mesh→RB transforms saved in data/RmatrixAndT/")


# 4) Load RB → WORLD transforms (Motive)
motive_tibia = pd.read_csv("../../data/csv_files/tibiaTransform.csv")
motive_femur = pd.read_csv("../../data/csv_files/femurTransform.csv")

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

pd.DataFrame(mesh_world_tibia, columns=headers).to_csv("../../data/csv_files/tibiaTransformForUnity.csv", index=False)
pd.DataFrame(mesh_world_femur, columns=headers).to_csv("../../data/csv_files/femurTransformForUnity.csv", index=False)

print("CSV for Unity written.")

#MARKERS FOR UNITY (ONLY POINTS PRESENT ON MESH)
tibia_markers_unity = np.zeros((n_frames, 9))   # AMT, LT, MT → 3x3
femur_markers_unity = np.zeros((n_frames, 9))   # AF,  LF, MF → 3x3

for i in range(n_frames):

    # ---------- TIBIA (AMT, LT, MT) ----------
    qw = mesh_world_tibia[i, 4]
    qx, qy, qz = mesh_world_tibia[i, 1], mesh_world_tibia[i, 2], mesh_world_tibia[i, 3]
    R_world = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T_world = mesh_world_tibia[i, 5:8].reshape(3, 1)

    AMT_mesh = meshtibia_local_unity[:, [0]]
    LT_mesh  = meshtibia_local_unity[:, [1]]
    MT_mesh  = meshtibia_local_unity[:, [2]]

    P_AMT = (R_world @ AMT_mesh + T_world).ravel()
    P_LT  = (R_world @ LT_mesh  + T_world).ravel()
    P_MT  = (R_world @ MT_mesh  + T_world).ravel()

    tibia_markers_unity[i, :] = np.concatenate([P_AMT, P_LT, P_MT])

    # ---------- FEMUR (AF, LF, MF) ----------
    qw = mesh_world_femur[i, 4]
    qx, qy, qz = mesh_world_femur[i, 1], mesh_world_femur[i, 2], mesh_world_femur[i, 3]
    R_world = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T_world = mesh_world_femur[i, 5:8].reshape(3, 1)

    AF_mesh = meshfemur_local_unity[:, [0]]
    LF_mesh = meshfemur_local_unity[:, [1]]
    MF_mesh = meshfemur_local_unity[:, [2]]

    P_AF = (R_world @ AF_mesh + T_world).ravel()
    P_LF = (R_world @ LF_mesh + T_world).ravel()
    P_MF = (R_world @ MF_mesh + T_world).ravel()

    femur_markers_unity[i, :] = np.concatenate([P_AF, P_LF, P_MF])
    
# HEADERS + CSV WRITE
tibia_cols = [
    "tibiaAMT_X","tibiaAMT_Y","tibiaAMT_Z",
    "tibiaLT_X", "tibiaLT_Y", "tibiaLT_Z",
    "tibiaMT_X", "tibiaMT_Y", "tibiaMT_Z",
]

femur_cols = [
    "femurAF_X","femurAF_Y","femurAF_Z",
    "femurLF_X","femurLF_Y","femurLF_Z",
    "femurMF_X","femurMF_Y","femurMF_Z",
]

tibia_markers_table = pd.DataFrame(tibia_markers_unity, columns=tibia_cols)
femur_markers_table = pd.DataFrame(femur_markers_unity, columns=femur_cols)

# MATLAB addvars(...,'Before',1,'Frame')
if "Frame" in motive_tibia.columns:
    tibia_markers_table.insert(0, "Frame", motive_tibia["Frame"].to_numpy())
else:
    tibia_markers_table.insert(0, "Frame", np.arange(n_frames))

if "Frame" in motive_femur.columns:
    femur_markers_table.insert(0, "Frame", motive_femur["Frame"].to_numpy())
else:
    femur_markers_table.insert(0, "Frame", np.arange(n_frames))

tibia_markers_table.to_csv("../../data/csv_files/tibiaMarkersForUnity.csv", index=False)
femur_markers_table.to_csv("../../data/csv_files/femurMarkersForUnity.csv", index=False)

print("Markers for Unity written.")
