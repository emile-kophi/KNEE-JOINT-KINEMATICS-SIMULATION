import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from scripts.kabsch_algorithm import kabsch_rt

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

# MAIN PIPELINE FUNCTION

def Mesh_to_World_Pipeline(
    tibia_vp_csv: str | Path,
    femur_vp_csv: str | Path,
    tibia_transform_csv: str | Path,
    femur_transform_csv: str | Path,
    output_dir: str | Path,
    output_RT_dir: str | Path,
) -> None:
    """
    Complete pipeline:
    Unity mesh → RB local → Motive world → Unity CSV + markers
    """

    output_dir = Path(output_dir)
    output_RT_dir = Path(output_RT_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_RT_dir.mkdir(parents=True, exist_ok=True)

    # 1) UNITY MESH POINTS → coordinate correction
    C = np.array(
        [[-1, 0, 0],
         [ 0, 1, 0],
         [ 0, 0,-1]],
        dtype=float
    )

    meshtibia_local_unity = np.array([
        [-0.02015125,  0.4032193,   0.03225459],  # AMT
        [ 0.03259825,  0.01721638, -0.01066572],  # LT
        [-0.03612585,  0.02022061,  0.01776088],  # MT
    ], dtype=float).T

    meshfemur_local_unity = np.array([
        [-0.002581612, -0.4347238,   0.04451473],   # AF
        [ 0.03755924,  -0.001844588, -0.01695582],  # LF
        [-0.03962225,  -0.004840371,  0.01212757],  # MF
    ], dtype=float).T

    meshtibia_local = C @ meshtibia_local_unity
    meshfemur_local = C @ meshfemur_local_unity

    # 2) RB-LOCAL VIRTUAL POINTS
    tibia_vp = pd.read_csv(tibia_vp_csv)
    femur_vp = pd.read_csv(femur_vp_csv)

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

    # 3) KABSCH: mesh → RB
    R_mesh_rb_tibia, T_mesh_rb_tibia = kabsch_rt(meshtibia_local, tibia_points_rb)
    R_mesh_rb_femur, T_mesh_rb_femur = kabsch_rt(meshfemur_local, femur_points_rb)

    np.savez(output_RT_dir / "mesh_to_rb_tibia.npz", R=R_mesh_rb_tibia, T=T_mesh_rb_tibia)
    np.savez(output_RT_dir / "mesh_to_rb_femur.npz", R=R_mesh_rb_femur, T=T_mesh_rb_femur)

    # 4) RB → WORLD (Motive)
    motive_tibia = pd.read_csv(tibia_transform_csv)
    motive_femur = pd.read_csv(femur_transform_csv)

    n_frames = len(motive_tibia)

    rb_world_tibia_wxyz = motive_tibia[
        ["Tibia_Rotation_W","Tibia_Rotation_X","Tibia_Rotation_Y","Tibia_Rotation_Z"]
    ].to_numpy()

    rb_world_femur_wxyz = motive_femur[
        ["Femur_Rotation_W","Femur_Rotation_X","Femur_Rotation_Y","Femur_Rotation_Z"]
    ].to_numpy()

    translation_tibia = motive_tibia[
        ["Tibia_Position_X","Tibia_Position_Y","Tibia_Position_Z"]
    ].to_numpy().T / 1000.0

    translation_femur = motive_femur[
        ["Femur_Position_X","Femur_Position_Y","Femur_Position_Z"]
    ].to_numpy().T / 1000.0

    # 5) mesh → world transforms (Unity)
    mesh_world_tibia = np.zeros((n_frames, 8))
    mesh_world_femur = np.zeros((n_frames, 8))

    for i in range(n_frames):

        # --- Tibia ---
        R_rb_world = Converter_Q_to_R(rb_world_tibia_wxyz[i])
        T_rb_world = translation_tibia[:, i].reshape(3,1)

        R_world = R_rb_world @ R_mesh_rb_tibia
        T_world = R_rb_world @ T_mesh_rb_tibia + T_rb_world

        q = Converter_R_to_QmatlabStyle(R_world)
        mesh_world_tibia[i,:] = [i, q[1], q[2], q[3], q[0], *T_world.ravel()]

        # --- Femur ---
        R_rb_world = Converter_Q_to_R(rb_world_femur_wxyz[i])
        T_rb_world = translation_femur[:, i].reshape(3,1)

        R_world = R_rb_world @ R_mesh_rb_femur
        T_world = R_rb_world @ T_mesh_rb_femur + T_rb_world

        q = Converter_R_to_QmatlabStyle(R_world)
        mesh_world_femur[i,:] = [i, q[1], q[2], q[3], q[0], *T_world.ravel()]

    headers = ["Frame","qx","qy","qz","qw","tx","ty","tz"]

    pd.DataFrame(mesh_world_tibia, columns=headers).to_csv(
        output_dir / "tibiaTransformForUnity.csv", index=False
    )
    pd.DataFrame(mesh_world_femur, columns=headers).to_csv(
        output_dir / "femurTransformForUnity.csv", index=False
    )

    # 6) MARKERS FOR UNITY
    tibia_markers = np.zeros((n_frames, 9))
    femur_markers = np.zeros((n_frames, 9))

    for i in range(n_frames):

        Rw = R.from_quat(mesh_world_tibia[i,1:5][[0,1,2,3]]).as_matrix()
        Tw = mesh_world_tibia[i,5:8].reshape(3,1)

        for j, P in enumerate(meshtibia_local_unity.T):
            tibia_markers[i, 3*j:3*j+3] = (Rw @ P.reshape(3,1) + Tw).ravel()

        Rw = R.from_quat(mesh_world_femur[i,1:5][[0,1,2,3]]).as_matrix()
        Tw = mesh_world_femur[i,5:8].reshape(3,1)

        for j, P in enumerate(meshfemur_local_unity.T):
            femur_markers[i, 3*j:3*j+3] = (Rw @ P.reshape(3,1) + Tw).ravel()

    tibia_cols = [
        "tibiaAMT_X","tibiaAMT_Y","tibiaAMT_Z",
        "tibiaLT_X","tibiaLT_Y","tibiaLT_Z",
        "tibiaMT_X","tibiaMT_Y","tibiaMT_Z",
    ]

    femur_cols = [
        "femurAF_X","femurAF_Y","femurAF_Z",
        "femurLF_X","femurLF_Y","femurLF_Z",
        "femurMF_X","femurMF_Y","femurMF_Z",
    ]

    tibia_df = pd.DataFrame(tibia_markers, columns=tibia_cols)
    femur_df = pd.DataFrame(femur_markers, columns=femur_cols)

    tibia_df.insert(0, "Frame", motive_tibia.get("Frame", np.arange(n_frames)))
    femur_df.insert(0, "Frame", motive_femur.get("Frame", np.arange(n_frames)))

    tibia_df.to_csv(output_dir / "tibiaMarkersForUnity.csv", index=False)
    femur_df.to_csv(output_dir / "femurMarkersForUnity.csv", index=False)

    print("Mesh → World pipeline completed successfully.")
