import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

# ==================================================
# 2) LOAD RB → WORLD TRANSFORMS (Motive)
# ==================================================
tibia_transform = pd.read_csv("../../data/csv_files/tibiaTransform.csv")
femur_transform = pd.read_csv("../../data/csv_files/femurTransform.csv")

# MATLAB: quat2rotm expects [qw qx qy qz]
# CSV order: qx qy qz qw  → reorder
tibia_q = tibia_transform.iloc[:, [4, 1, 2, 3]].to_numpy()
femur_q = femur_transform.iloc[:, [4, 1, 2, 3]].to_numpy()

tibia_rotm = R.from_quat(
    np.column_stack([tibia_q[:,1], tibia_q[:,2], tibia_q[:,3], tibia_q[:,0]])
).as_matrix()

femur_rotm = R.from_quat(
    np.column_stack([femur_q[:,1], femur_q[:,2], femur_q[:,3], femur_q[:,0]])
).as_matrix()

tibia_xyz = tibia_transform.iloc[:, [5, 6, 7]].to_numpy()
femur_xyz = femur_transform.iloc[:, [5, 6, 7]].to_numpy()

# ==================================================
# 3) LOAD LOCAL RB POINTS (virtual points)
# ==================================================
tibia_vp = pd.read_csv("../../data/csv_files/tibiaVP.csv")
femur_vp = pd.read_csv("../../data/csv_files/femurVP.csv")

# Tibia locals
ALT_local = tibia_vp.iloc[0, [0, 1, 2]].to_numpy().reshape(3,1)
AMT_local = tibia_vp.iloc[0, [3, 4, 5]].to_numpy().reshape(3,1)
LT_local  = tibia_vp.iloc[0, [6, 7, 8]].to_numpy().reshape(3,1)
MT_local  = tibia_vp.iloc[0, [9,10,11]].to_numpy().reshape(3,1)

# Femur locals
AF_local = femur_vp.iloc[0, [0, 1, 2]].to_numpy().reshape(3,1)
LF_local = femur_vp.iloc[0, [3, 4, 5]].to_numpy().reshape(3,1)
MF_local = femur_vp.iloc[0, [6, 7, 8]].to_numpy().reshape(3,1)

# ==================================================
# 4) GLOBAL POINT RECONSTRUCTION
# ==================================================
n_frames = tibia_xyz.shape[0]

tibia_global = np.zeros((n_frames, 12))
femur_global = np.zeros((n_frames, 9))

for i in range(n_frames):

    R_t = tibia_rotm[i]
    T_t = tibia_xyz[i].reshape(3,1)

    tibia_global[i, 0:3]   = (R_t @ ALT_local + T_t).ravel()
    tibia_global[i, 3:6]   = (R_t @ AMT_local + T_t).ravel()
    tibia_global[i, 6:9]   = (R_t @ LT_local  + T_t).ravel()
    tibia_global[i, 9:12]  = (R_t @ MT_local  + T_t).ravel()

    R_f = femur_rotm[i]
    T_f = femur_xyz[i].reshape(3,1)

    femur_global[i, 0:3] = (R_f @ AF_local + T_f).ravel()
    femur_global[i, 3:6] = (R_f @ LF_local + T_f).ravel()
    femur_global[i, 6:9] = (R_f @ MF_local + T_f).ravel()

# MATLAB division by 1000 (mm → m)
tibia_global /= 1000
femur_global /= 1000

# ==================================================
# WRITE CSV (VALIDATION)
# ==================================================
tibia_cols = [
    "tibiaALT_X","tibiaALT_Y","tibiaALT_Z",
    "tibiaAMT_X","tibiaAMT_Y","tibiaAMT_Z",
    "tibiaLT_X","tibiaLT_Y","tibiaLT_Z",
    "tibiaMT_X","tibiaMT_Y","tibiaMT_Z"
]

femur_cols = [
    "femurAF_X","femurAF_Y","femurAF_Z",
    "femurLF_X","femurLF_Y","femurLF_Z",
    "femurMF_X","femurMF_Y","femurMF_Z"
]

pd.DataFrame(tibia_global, columns=tibia_cols).to_csv(
     "../../data/csv_files/tibiaGlobalPoints.csv", index=False
)

pd.DataFrame(femur_global, columns=femur_cols).to_csv(
      "../../data/csv_files/femurGlobalPoints.csv", index=False
)

print("Global points generated successfully.")
