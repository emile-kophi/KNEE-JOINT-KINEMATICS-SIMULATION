import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from .kabsch_algorithm import kabsch_rt

# QUATERNION / ROTATION HELPERS

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

# MAIN PIPELINE Unity mesh → RB local → Motive world → Unity CSV + RT matrices
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

    # CONSTANTS
    # Motive RH → Unity LH
    S = np.diag([1.0, 1.0, -1.0])

    # 180° rotation around Y (Blender → Motive)
    Rot_B2M = np.diag([-1.0, 1.0, -1.0])

    # MESH LOCAL POINTS (BLENDER LOCAL, meters)

    # FEMUR
    meshfemur_local_blender = np.array([
        [-0.01759300,   0.2254449,   0.02862103],   # AF
        [ 0.02254823,  -0.2074343,  -0.03284946],   # LF
        [-0.05463326,  -0.2044385,  -0.00376609],   # MF
    ], dtype=float).T

    # PATELLA
    meshpatella_local_blender = np.array([
        [ 0.00104400,  -0.02765099,  0.005901016],  # PP
        [-0.01865400,  -0.00665200, -0.008743007],  # MP
        [ 0.01838301,  -0.01326200,  0.01198900],   # LP
    ], dtype=float).T

    # TIBIA
    meshtibia_local_blender = np.array([
        [-0.02177037,  -0.2164885,   0.01993167],   # AMT
        [ 0.03097912,   0.1695145,  -0.02298864],   # LT
        [-0.03774497,   0.1665102,   0.005437955],  # MT
    ], dtype=float).T

    # Apply Blender → Motive rotation
    meshfemur_local   = Rot_B2M @ meshfemur_local_blender
    meshtibia_local   = Rot_B2M @ meshtibia_local_blender
    meshpatella_local = Rot_B2M @ meshpatella_local_blender

    # RB-LOCAL VIRTUAL POINTS (FROM MOTIVE VP, meters)
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

    # KABSCH: MESH → RB
    R_mesh_rb_tibia,   T_mesh_rb_tibia   = kabsch_rt(meshtibia_local,   tibia_points_rb)
    R_mesh_rb_femur,   T_mesh_rb_femur   = kabsch_rt(meshfemur_local,   femur_points_rb)
    R_mesh_rb_patella, T_mesh_rb_patella = kabsch_rt(meshpatella_local, patella_points_rb)

    np.savez(output_RT_dir / "mesh_to_rb_tibia.npz",   R=R_mesh_rb_tibia,   T=T_mesh_rb_tibia)
    np.savez(output_RT_dir / "mesh_to_rb_femur.npz",   R=R_mesh_rb_femur,   T=T_mesh_rb_femur)
    np.savez(output_RT_dir / "mesh_to_rb_patella.npz", R=R_mesh_rb_patella, T=T_mesh_rb_patella)

    # LOAD RB → WORLD (Motive)
    Tt = pd.read_csv(tibia_transform_csv)
    Tf = pd.read_csv(femur_transform_csv)
    Tp = pd.read_csv(patella_transform_csv)

    n_frames = min(len(Tt), len(Tf), len(Tp))

    rb_world_tibia_wxyz   = Tt[["Tibia_Rotation_W","Tibia_Rotation_X","Tibia_Rotation_Y","Tibia_Rotation_Z"]].to_numpy()
    rb_world_femur_wxyz   = Tf[["Femur_Rotation_W","Femur_Rotation_X","Femur_Rotation_Y","Femur_Rotation_Z"]].to_numpy()
    rb_world_patella_wxyz = Tp[["Patella_Rotation_W","Patella_Rotation_X","Patella_Rotation_Y","Patella_Rotation_Z"]].to_numpy()

    translation_tibia   = Tt[["Tibia_Position_X","Tibia_Position_Y","Tibia_Position_Z"]].to_numpy().T / 1000.0
    translation_femur   = Tf[["Femur_Position_X","Femur_Position_Y","Femur_Position_Z"]].to_numpy().T / 1000.0
    translation_patella = Tp[["Patella_Position_X","Patella_Position_Y","Patella_Position_Z"]].to_numpy().T / 1000.0


    mesh_world_tibia   = np.zeros((n_frames, 8))
    mesh_world_femur   = np.zeros((n_frames, 8))
    mesh_world_patella = np.zeros((n_frames, 8))

    R_world_tibia_all   = np.zeros((n_frames, 3, 3))
    T_world_tibia_all   = np.zeros((n_frames, 3))

    R_world_femur_all   = np.zeros((n_frames, 3, 3))
    T_world_femur_all   = np.zeros((n_frames, 3))

    R_world_patella_all = np.zeros((n_frames, 3, 3))
    T_world_patella_all = np.zeros((n_frames, 3))


    for i in range(n_frames):

        #  TIBIA 
        R_rb = Converter_Q_to_R(rb_world_tibia_wxyz[i])
        T_rb = translation_tibia[:, i].reshape(3,1)

        R_tmp = R_rb @ R_mesh_rb_tibia
        T_tmp = R_rb @ T_mesh_rb_tibia + T_rb

        R_world = S @ R_tmp @ S
        T_world = S @ T_tmp

        R_world_tibia_all[i] = R_world
        T_world_tibia_all[i] = T_world.ravel()

        q = Converter_R_to_QmatlabStyle(R_world)
        mesh_world_tibia[i] = [i, q[1], q[2], q[3], q[0], *T_world.ravel()]

        #  FEMUR 
        R_rb = Converter_Q_to_R(rb_world_femur_wxyz[i])
        T_rb = translation_femur[:, i].reshape(3,1)

        R_tmp = R_rb @ R_mesh_rb_femur
        T_tmp = R_rb @ T_mesh_rb_femur + T_rb

        R_world = S @ R_tmp @ S
        T_world = S @ T_tmp

        R_world_femur_all[i] = R_world
        T_world_femur_all[i] = T_world.ravel()

        q = Converter_R_to_QmatlabStyle(R_world)
        mesh_world_femur[i] = [i, q[1], q[2], q[3], q[0], *T_world.ravel()]

        #  PATELLA 
        R_rb = Converter_Q_to_R(rb_world_patella_wxyz[i])
        T_rb = translation_patella[:, i].reshape(3,1)

        R_tmp = R_rb @ R_mesh_rb_patella
        T_tmp = R_rb @ T_mesh_rb_patella + T_rb

        R_world = S @ R_tmp @ S
        T_world = S @ T_tmp

        R_world_patella_all[i] = R_world
        T_world_patella_all[i] = T_world.ravel()

        q = Converter_R_to_QmatlabStyle(R_world)
        mesh_world_patella[i] = [i, q[1], q[2], q[3], q[0], *T_world.ravel()]

    # SAVE WORLD RT MATRICES
    np.savez(output_RT_dir / "mesh_to_world_tibia.npz",
             R=R_world_tibia_all, T=T_world_tibia_all)

    np.savez(output_RT_dir / "mesh_to_world_femur.npz",
             R=R_world_femur_all, T=T_world_femur_all)

    np.savez(output_RT_dir / "mesh_to_world_patella.npz",
             R=R_world_patella_all, T=T_world_patella_all)

    # WRITE CSV FOR UNITY
    headers = ["Frame","qx","qy","qz","qw","tx","ty","tz"]

    pd.DataFrame(mesh_world_tibia,   columns=headers).to_csv(output_dir / "tibiaTransformForUnity.csv",   index=False)
    pd.DataFrame(mesh_world_femur,   columns=headers).to_csv(output_dir / "femurTransformForUnity.csv",   index=False)
    pd.DataFrame(mesh_world_patella, columns=headers).to_csv(output_dir / "patellaTransformForUnity.csv", index=False)

    print("Mesh → World pipeline completed successfully.")
