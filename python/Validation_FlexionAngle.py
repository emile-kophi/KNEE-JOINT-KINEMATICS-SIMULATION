import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pathlib import Path

# Conversion matrix Unity → Motive-like
C = np.array([
    [-1, 0,  0],
    [ 0, 1,  0],
    [ 0, 0, -1],
], dtype=float)

# LOAD mesh → RB TRANSFORMS
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" / "RmatrixAndT"

# Tibia
tibia_data = np.load(DATA_DIR / "mesh_to_rb_tibia.npz")
R_mesh_rb_tibia = tibia_data["R"]                 # (3,3)
T_mesh_rb_tibia = tibia_data["T"].ravel()         # (3,)

# Femur
femur_data = np.load(DATA_DIR / "mesh_to_rb_femur.npz")
R_mesh_rb_femur = femur_data["R"]                 # (3,3)
T_mesh_rb_femur = femur_data["T"].ravel()         # (3,)


# 1) ANATOMICAL LOCAL POINTS (Unity local)

# TIBIA
TibiaProx_Unity = np.array([0.0008561252,  0.09804206, 0.01295095])
TibiaDist_Unity = np.array([0.003904605,   0.3762208,  0.01639927])

# FEMUR
FemurProx_Unity = np.array([0.03055387,  -0.3575693,  0.0149763])
FemurDist_Unity = np.array([0.004923976, -0.04483421, 0.01108846])

# FEMUR CONDYLES
LFlocal_Unity = np.array([ 0.03755924,  -0.001844588, -0.01695582])
MFlocal_Unity = np.array([-0.03962225,  -0.004840371,  0.01212757])

# Convert to Blender-like coordinates
Tibia_Prox = C @ TibiaProx_Unity
Tibia_Dist = C @ TibiaDist_Unity
Femur_Prox = C @ FemurProx_Unity
Femur_Dist = C @ FemurDist_Unity
LF_local   = C @ LFlocal_Unity
MF_local   = C @ MFlocal_Unity

# LOAD RB → WORLD TRANSFORMS (Motive)
Rb_t = pd.read_csv("../data/csv_files/tibiaTransform.csv")
Rb_f = pd.read_csv("../data/csv_files/femurTransform.csv")

n_frames = len(Rb_t)

q_rb_t = np.column_stack([
    Rb_t["Tibia_Rotation_W"],
    Rb_t["Tibia_Rotation_X"],
    Rb_t["Tibia_Rotation_Y"],
    Rb_t["Tibia_Rotation_Z"],
])

q_rb_f = np.column_stack([
    Rb_f["Femur_Rotation_W"],
    Rb_f["Femur_Rotation_X"],
    Rb_f["Femur_Rotation_Y"],
    Rb_f["Femur_Rotation_Z"],
])

T_rb_t = np.vstack([
    Rb_t["Tibia_Position_X"],
    Rb_t["Tibia_Position_Y"],
    Rb_t["Tibia_Position_Z"],
]) / 1000.0   # mm → m

T_rb_f = np.vstack([
    Rb_f["Femur_Position_X"],
    Rb_f["Femur_Position_Y"],
    Rb_f["Femur_Position_Z"],
]) / 1000.0   # mm → m

# FLEXION ANGLE VECTOR
theta_deg = np.zeros(n_frames)

# Helper: Motive quaternion [w x y z] → rotation matrix
def quat_wxyz_to_rotm(q_wxyz):
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    return R.from_quat(q_xyzw).as_matrix()

# 5) MAIN LOOP
for i in range(n_frames):

    # TIBIA: mesh → RB → world
    R_rb_tibia = quat_wxyz_to_rotm(q_rb_t[i])
    T_rb_tibia = T_rb_t[:, i]

    Tibia_Prox_world = (
        R_rb_tibia @ (R_mesh_rb_tibia @ Tibia_Prox + T_mesh_rb_tibia)
        + T_rb_tibia
    )

    Tibia_Dist_world = (
        R_rb_tibia @ (R_mesh_rb_tibia @ Tibia_Dist + T_mesh_rb_tibia)
        + T_rb_tibia
    )

    axis_tibia = Tibia_Dist_world - Tibia_Prox_world
    axis_tibia /= np.linalg.norm(axis_tibia)

    # FEMUR: mesh → RB → world
    R_rb_femur = quat_wxyz_to_rotm(q_rb_f[i])
    T_rb_femur = T_rb_f[:, i]

    Femur_Prox_world = (
        R_rb_femur @ (R_mesh_rb_femur @ Femur_Prox + T_mesh_rb_femur)
        + T_rb_femur
    )

    Femur_Dist_world = (
        R_rb_femur @ (R_mesh_rb_femur @ Femur_Dist + T_mesh_rb_femur)
        + T_rb_femur
    )

    axis_femur = Femur_Dist_world - Femur_Prox_world
    axis_femur /= np.linalg.norm(axis_femur)

    # FEMUR CONDYLES AXIS
    LF_world = (
        R_rb_femur @ (R_mesh_rb_femur @ LF_local + T_mesh_rb_femur)
        + T_rb_femur
    )

    MF_world = (
        R_rb_femur @ (R_mesh_rb_femur @ MF_local + T_mesh_rb_femur)
        + T_rb_femur
    )

    condyle_axis = LF_world - MF_world
    condyle_axis /= np.linalg.norm(condyle_axis)

    # PROJECT TIBIA AXIS ON FEMUR SAGITTAL PLANE
    t = axis_tibia
    c = condyle_axis

    # remove lateral component: t_proj = t - (t·c)c
    t_proj = t - np.dot(t, c) * c
    t_proj /= np.linalg.norm(t_proj)

    # ANATOMICAL FLEXION ANGLE
    f = axis_femur
    cosang = np.dot(f, t_proj)
    cosang = np.clip(cosang, -1.0, 1.0)

    theta_deg[i] = np.degrees(np.arccos(cosang))

#  RESULTS
plt.figure()
plt.plot(theta_deg, linewidth=2)
plt.xlabel("Frame")
plt.ylabel("Flexion angle (deg)")
plt.title("Knee Flexion Angle (Anatomic)")
plt.grid(True)
plt.show()

print(f"Flessione min: {np.min(theta_deg):.2f}°")
print(f"Flessione max: {np.max(theta_deg):.2f}°")
