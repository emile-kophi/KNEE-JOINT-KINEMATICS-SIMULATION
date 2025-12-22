import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from .kabsch_algorithm import kabsch_rt

# Quaternion / rotation helpers
def Converter_R_to_QmatlabStyle(rotm: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to MATLAB-style quaternion [w x y z].
    """
    q_xyzw = R.from_matrix(rotm).as_quat()  # [x y z w]
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)


def Converter_Q_to_R(q_wxyz: np.ndarray) -> np.ndarray:
    """
    Convert Motive/MATLAB quaternion [w x y z] to rotation matrix.
    """
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=float)
    return R.from_quat(q_xyzw).as_matrix()

# MAIN PIPELINE
def Mesh_to_World_Pipeline(
    tibia_vp_csv: str | Path,
    femur_vp_csv: str | Path,
    patella_vp_csv: str | Path,
    tibia_transform_csv: str | Path,
    femur_transform_csv: str | Path,
    patella_transform_csv: str | Path,
    output_dir: str | Path,
    output_RT_dir: str | Path,
) -> None:
    """
    Complete pipeline:
    Unity mesh → RB local → Motive world → Unity CSV + RT matrices
    """
    # PATH NORMALIZATION
    tibia_vp_csv = Path(tibia_vp_csv)
    femur_vp_csv = Path(femur_vp_csv)
    patella_vp_csv = Path(patella_vp_csv)
    tibia_transform_csv = Path(tibia_transform_csv)
    femur_transform_csv = Path(femur_transform_csv)
    patella_transform_csv = Path(patella_transform_csv)

    output_dir = Path(output_dir)
    output_RT_dir = Path(output_RT_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_RT_dir.mkdir(parents=True, exist_ok=True)

    # UNITY MESH POINTS (local)
    C = np.array(
        [[-1, 0, 0],
         [ 0, 1, 0],
         [ 0, 0,-1]],
        dtype=float
    )

#     meshtibia_local_unity = np.array([
#         [-0.02015125,  0.4032193,   0.03225459],  # AMT
#         [ 0.03259825,  0.01721638, -0.01066572],  # LT
#         [-0.03612585,  0.02022061,  0.01776088],  # MT
#     ], dtype=float).T

#     meshfemur_local_unity = np.array([
#         [-0.002581612, -0.4347238,   0.04451473],   # AF
#         [ 0.03755924,  -0.001844588, -0.01695582],  # LF
#         [-0.03962225,  -0.004840371,  0.01212757],  # MF
#     ], dtype=float).T
#     meshpatella_local_unity = np.array([
#     [0.001044095, -0.02765156, -0.005900663],  # PP 
#    [-0.01865382, -0.006652482,  0.008743234],  # MP 
#     [0.01838263, -0.01326214, -0.01198923],    # LP 
# ], dtype=float).T
       # FEMUR (AF = origin)
    meshfemur_local_unity = np.array([
         [0.0,          0.0,         0.0],          # AF
         [0.040140872,  0.432879289, -0.061470632], # LF
         [-0.037040648, 0.429883489, -0.032387192], # MF
    ], dtype=float).T


# PATELLA
    meshpatella_local_unity = np.array([
    [ 0.006379797,  0.4206465,  -0.04991192],  # PP
    [-0.01331811,   0.4416456,  -0.03526796],  # MP
    [ 0.0237183,    0.4350359,  -0.05600054],  # LP
    ], dtype=float).T


# TIBIA
    meshtibia_local_unity = np.array([
    [-0.009806024,  0.8415409,  -0.01789982],  # AMT
    [ 0.04294349,   0.4555378,  -0.06081994],  # LT
    [-0.02578063,   0.458542,   -0.03239334],  # MT
    ], dtype=float).T



    meshtibia_local = C @ meshtibia_local_unity
    meshfemur_local = C @ meshfemur_local_unity
    meshpatella_local = C @ meshpatella_local_unity

    # RB-LOCAL VIRTUAL POINTS

    tibia_vp = pd.read_csv(tibia_vp_csv)
    femur_vp = pd.read_csv(femur_vp_csv)
    patella_vp = pd.read_csv(patella_vp_csv)

    tibia_points_rb = np.array([
        tibia_vp.loc[0, ["tibiaAMTlocal_X","tibiaAMTlocal_Y","tibiaAMTlocal_Z"]],
        tibia_vp.loc[0, ["tibiaLTlocal_X","tibiaLTlocal_Y","tibiaLTlocal_Z"]],
        tibia_vp.loc[0, ["tibiaMTlocal_X","tibiaMTlocal_Y","tibiaMTlocal_Z"]],
    ], dtype=float).T / 1000.0

    femur_points_rb = np.array([
        femur_vp.loc[0, ["femurAFlocal_X","femurAFlocal_Y","femurAFlocal_Z"]],
        femur_vp.loc[0, ["femurLFlocal_X","femurLFlocal_Y","femurLFlocal_Z"]],
        femur_vp.loc[0, ["femurMFlocal_X","femurMFlocal_Y","femurMFlocal_Z"]],
    ], dtype=float).T / 1000.0
    patella_points_rb = np.array([
        patella_vp.loc[0, ["patellaPPlocal_X","patellaPPlocal_Y","patellaPPlocal_Z"]],
        patella_vp.loc[0, ["patellaMPlocal_X","patellaMPlocal_Y","patellaMPlocal_Z"]],
        patella_vp.loc[0, ["patellaLPlocal_X","patellaLPlocal_Y","patellaLPlocal_Z"]],
    ], dtype=float).T / 1000.0

    # KABSCH: mesh → RB
    R_mesh_rb_tibia, T_mesh_rb_tibia = kabsch_rt(meshtibia_local, tibia_points_rb)
    R_mesh_rb_femur, T_mesh_rb_femur = kabsch_rt(meshfemur_local, femur_points_rb)
    R_mesh_rb_patella, T_mesh_rb_patella = kabsch_rt(meshpatella_local, patella_points_rb)

    np.savez(output_RT_dir / "mesh_to_rb_tibia.npz", R=R_mesh_rb_tibia, T=T_mesh_rb_tibia)
    np.savez(output_RT_dir / "mesh_to_rb_femur.npz", R=R_mesh_rb_femur, T=T_mesh_rb_femur)
    np.savez(output_RT_dir / "mesh_to_rb_patella.npz", R=R_mesh_rb_patella, T=T_mesh_rb_patella)

    # RB → WORLD (Motive)
    motive_tibia = pd.read_csv(tibia_transform_csv)
    motive_femur = pd.read_csv(femur_transform_csv)
    motive_patella = pd.read_csv(patella_transform_csv)

    n_frames = len(motive_tibia)

    rb_world_tibia_wxyz = motive_tibia[
        ["Tibia_Rotation_W","Tibia_Rotation_X","Tibia_Rotation_Y","Tibia_Rotation_Z"]
    ].to_numpy()

    rb_world_femur_wxyz = motive_femur[
        ["Femur_Rotation_W","Femur_Rotation_X","Femur_Rotation_Y","Femur_Rotation_Z"]
    ].to_numpy()

    rb_world_patella_wxyz = motive_patella[
        ["Patella_Rotation_W","Patella_Rotation_X","Patella_Rotation_Y","Patella_Rotation_Z"]
    ].to_numpy()

    translation_tibia = motive_tibia[
        ["Tibia_Position_X","Tibia_Position_Y","Tibia_Position_Z"]
    ].to_numpy().T / 1000.0

    translation_femur = motive_femur[
        ["Femur_Position_X","Femur_Position_Y","Femur_Position_Z"]
    ].to_numpy().T / 1000.0

    translation_patella = motive_patella[
        ["Patella_Position_X","Patella_Position_Y","Patella_Position_Z"]
    ].to_numpy().T / 1000.0

    #  MESH → WORLD (ALL FRAMES)
    mesh_world_tibia = np.zeros((n_frames, 8))
    mesh_world_femur = np.zeros((n_frames, 8))
    mesh_world_patella = np.zeros((n_frames, 8))

    R_world_tibia_all = np.zeros((n_frames, 3, 3))
    T_world_tibia_all = np.zeros((n_frames, 3))

    R_world_femur_all = np.zeros((n_frames, 3, 3))
    T_world_femur_all = np.zeros((n_frames, 3))

    R_world_patella_all = np.zeros((n_frames, 3, 3))
    T_world_patella_all = np.zeros((n_frames, 3))

    for i in range(n_frames):

        # -------- TIBIA --------
        R_rb_world = Converter_Q_to_R(rb_world_tibia_wxyz[i])
        T_rb_world = translation_tibia[:, i].reshape(3,1)

        R_world = R_rb_world @ R_mesh_rb_tibia
        T_world = R_rb_world @ T_mesh_rb_tibia + T_rb_world

        R_world_tibia_all[i] = R_world
        T_world_tibia_all[i] = T_world.ravel()

        q = Converter_R_to_QmatlabStyle(R_world)
        mesh_world_tibia[i,:] = [i, q[1], q[2], q[3], q[0], *T_world.ravel()]

        # -------- FEMUR --------
        R_rb_world = Converter_Q_to_R(rb_world_femur_wxyz[i])
        T_rb_world = translation_femur[:, i].reshape(3,1)

        R_world = R_rb_world @ R_mesh_rb_femur
        T_world = R_rb_world @ T_mesh_rb_femur + T_rb_world

        R_world_femur_all[i] = R_world
        T_world_femur_all[i] = T_world.ravel()

        q = Converter_R_to_QmatlabStyle(R_world)
        mesh_world_femur[i,:] = [i, q[1], q[2], q[3], q[0], *T_world.ravel()]

        # -------- PATELLA --------
        R_rb_world = Converter_Q_to_R(rb_world_patella_wxyz[i])
        T_rb_world = translation_patella[:, i].reshape(3,1)

        R_world = R_rb_world @ R_mesh_rb_patella
        T_world = R_rb_world @ T_mesh_rb_patella + T_rb_world

        R_world_patella_all[i] = R_world
        T_world_patella_all[i] = T_world.ravel()

        q = Converter_R_to_QmatlabStyle(R_world)
        mesh_world_patella[i,:] = [i, q[1], q[2], q[3], q[0], *T_world.ravel()]

    # SAVE RT MATRICES 
    np.savez(
        output_RT_dir / "mesh_to_world_tibia.npz",
        R=R_world_tibia_all,
        T=T_world_tibia_all
    )

    np.savez(
        output_RT_dir / "mesh_to_world_femur.npz",
        R=R_world_femur_all,
        T=T_world_femur_all
    )
    np.savez(
        output_RT_dir / "mesh_to_world_patella.npz",
        R=R_world_patella_all,
        T=T_world_patella_all
    )

    # SAVE UNITY CSV
    headers = ["Frame","qx","qy","qz","qw","tx","ty","tz"]

    pd.DataFrame(mesh_world_tibia, columns=headers).to_csv(
        output_dir / "tibiaTransformForUnity.csv", index=False
    )

    pd.DataFrame(mesh_world_femur, columns=headers).to_csv(
        output_dir / "femurTransformForUnity.csv", index=False
    )
    pd.DataFrame(mesh_world_patella, columns=headers).to_csv(
        output_dir / "patellaTransformForUnity.csv", index=False
    )
    print("Mesh → World pipeline completed successfully.")
