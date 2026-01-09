import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from .kabsch_algorithm import kabsch_rt

# QUATERNION / ROTATION HELPERS
def Converter_R_to_QmatlabStyle(rotm: np.ndarray) -> np.ndarray:
    q_xyzw = R.from_matrix(rotm).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)


def Converter_Q_to_R(q_wxyz: np.ndarray) -> np.ndarray:
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=float)
    return R.from_quat(q_xyzw).as_matrix()

# MESH → WORLD PIPELINE
def Mesh_to_World_Pipeline(
    tibia_vp_csv: str | Path,
    femur_vp_csv: str | Path,
    patella_vp_csv: str | Path,
    tibia_transform_csv: str | Path,
    femur_transform_csv: str | Path,
    patella_transform_csv: str | Path,
    output_dir: str | Path,
    output_RT_dir: str | Path,
    anatomy: dict,
) -> None:

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

    use_patella = patella_vp_csv.is_file() and patella_transform_csv.is_file()

    # UNITY CONVERSION
    S = np.diag([1.0, 1.0, -1.0])

    # MESH LOCAL LANDMARKS
    meshtibia_local = np.column_stack([
        anatomy["tibia"]["AMT"],
        anatomy["tibia"]["LT"],
        anatomy["tibia"]["MT"],
    ])

    meshfemur_local = np.column_stack([
        anatomy["femur"]["head"],
        anatomy["femur"]["LF"],
        anatomy["femur"]["MF"],
    ])

    meshpatella_local = np.column_stack([
        anatomy["patella"]["PP"],
        anatomy["patella"]["MP"],
        anatomy["patella"]["LP"],
    ])

    # LOAD VIRTUAL POINTS
    tibia_vp = pd.read_csv(tibia_vp_csv)
    femur_vp = pd.read_csv(femur_vp_csv)
    if use_patella:
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

    if use_patella:
        patella_points_rb = np.array([
            patella_vp.loc[0, ["patellaPPlocal_X","patellaPPlocal_Y","patellaPPlocal_Z"]],
            patella_vp.loc[0, ["patellaMPlocal_X","patellaMPlocal_Y","patellaMPlocal_Z"]],
            patella_vp.loc[0, ["patellaLPlocal_X","patellaLPlocal_Y","patellaLPlocal_Z"]],
        ], dtype=float).T / 1000.0

    # KABSCH: mesh → rigid body
    R_mesh_rb_tibia,   T_mesh_rb_tibia   = kabsch_rt(meshtibia_local, tibia_points_rb)
    R_mesh_rb_femur,   T_mesh_rb_femur   = kabsch_rt(meshfemur_local, femur_points_rb)
    if use_patella:
        R_mesh_rb_patella, T_mesh_rb_patella = kabsch_rt(meshpatella_local, patella_points_rb)

    np.savez(output_RT_dir / "mesh_to_rb_tibia.npz", R=R_mesh_rb_tibia, T=T_mesh_rb_tibia)
    np.savez(output_RT_dir / "mesh_to_rb_femur.npz", R=R_mesh_rb_femur, T=T_mesh_rb_femur)
    if use_patella:
        np.savez(output_RT_dir / "mesh_to_rb_patella.npz", R=R_mesh_rb_patella, T=T_mesh_rb_patella)

    # LOAD RIGID BODY TRANSFORMS (WORLD MOTIVE)
    Tt = pd.read_csv(tibia_transform_csv)
    Tf = pd.read_csv(femur_transform_csv)
    if use_patella:
        Tp = pd.read_csv(patella_transform_csv)

    n_frames = min(len(Tt), len(Tf)) if not use_patella else min(len(Tt), len(Tf), len(Tp))

    rb_world_tibia_wxyz = Tt[["Tibia_Rotation_W","Tibia_Rotation_X","Tibia_Rotation_Y","Tibia_Rotation_Z"]].to_numpy()
    rb_world_femur_wxyz = Tf[["Femur_Rotation_W","Femur_Rotation_X","Femur_Rotation_Y","Femur_Rotation_Z"]].to_numpy()
    if use_patella:
        rb_world_patella_wxyz = Tp[["Patella_Rotation_W","Patella_Rotation_X","Patella_Rotation_Y","Patella_Rotation_Z"]].to_numpy()

    translation_tibia = Tt[["Tibia_Position_X","Tibia_Position_Y","Tibia_Position_Z"]].to_numpy().T / 1000.0
    translation_femur = Tf[["Femur_Position_X","Femur_Position_Y","Femur_Position_Z"]].to_numpy().T / 1000.0
    if use_patella:
        translation_patella = Tp[["Patella_Position_X","Patella_Position_Y","Patella_Position_Z"]].to_numpy().T / 1000.0

    # OUTPUT STORAGE
    mesh_world_tibia = np.zeros((n_frames, 8))
    mesh_world_femur = np.zeros((n_frames, 8))
    if use_patella:
        mesh_world_patella = np.zeros((n_frames, 8))

    R_world_tibia_all = np.zeros((n_frames, 3, 3))
    T_world_tibia_all = np.zeros((n_frames, 3))
    R_world_femur_all = np.zeros((n_frames, 3, 3))
    T_world_femur_all = np.zeros((n_frames, 3))
    if use_patella:
        R_world_patella_all = np.zeros((n_frames, 3, 3))
        T_world_patella_all = np.zeros((n_frames, 3))

    # MAIN LOOP
    for i in range(n_frames):

        # ---------------- TIBIA ----------------
        R_rb = Converter_Q_to_R(rb_world_tibia_wxyz[i])
        T_rb = translation_tibia[:, i].reshape(3,1)

        R_tmp = R_rb @ R_mesh_rb_tibia
        T_tmp = R_rb @ T_mesh_rb_tibia + T_rb

        # SAVE WORLD MOTIVE (NO S)
        R_world_tibia_all[i] = R_tmp
        T_world_tibia_all[i] = T_tmp.ravel()

        # UNITY ONLY
        R_unity = S @ R_tmp @ S
        T_unity = S @ T_tmp

        q = Converter_R_to_QmatlabStyle(R_unity)
        mesh_world_tibia[i] = [i, q[1], q[2], q[3], q[0], *T_unity.ravel()]

        # ---------------- FEMUR ----------------
        R_rb = Converter_Q_to_R(rb_world_femur_wxyz[i])
        T_rb = translation_femur[:, i].reshape(3,1)

        R_tmp = R_rb @ R_mesh_rb_femur
        T_tmp = R_rb @ T_mesh_rb_femur + T_rb

        R_world_femur_all[i] = R_tmp
        T_world_femur_all[i] = T_tmp.ravel()

        R_unity = S @ R_tmp @ S
        T_unity = S @ T_tmp

        q = Converter_R_to_QmatlabStyle(R_unity)
        mesh_world_femur[i] = [i, q[1], q[2], q[3], q[0], *T_unity.ravel()]

        # ---------------- PATELLA ----------------
        if use_patella:
            R_rb = Converter_Q_to_R(rb_world_patella_wxyz[i])
            T_rb = translation_patella[:, i].reshape(3,1)

            R_tmp = R_rb @ R_mesh_rb_patella
            T_tmp = R_rb @ T_mesh_rb_patella + T_rb

            R_world_patella_all[i] = R_tmp
            T_world_patella_all[i] = T_tmp.ravel()

            R_unity = S @ R_tmp @ S
            T_unity = S @ T_tmp

            q = Converter_R_to_QmatlabStyle(R_unity)
            mesh_world_patella[i] = [i, q[1], q[2], q[3], q[0], *T_unity.ravel()]

    # SAVE NPZ (WORLD MOTIVE)
    np.savez(output_RT_dir / "mesh_to_world_tibia.npz", R=R_world_tibia_all, T=T_world_tibia_all)
    np.savez(output_RT_dir / "mesh_to_world_femur.npz", R=R_world_femur_all, T=T_world_femur_all)
    if use_patella:
        np.savez(output_RT_dir / "mesh_to_world_patella.npz", R=R_world_patella_all, T=T_world_patella_all)

    # SAVE UNITY CSV
    headers = ["Frame","qx","qy","qz","qw","tx","ty","tz"]

    pd.DataFrame(mesh_world_tibia, columns=headers).to_csv(
        output_dir / "tibiaTransformForUnity.csv", index=False
    )
    pd.DataFrame(mesh_world_femur, columns=headers).to_csv(
        output_dir / "femurTransformForUnity.csv", index=False
    )
    if use_patella:
        pd.DataFrame(mesh_world_patella, columns=headers).to_csv(
            output_dir / "patellaTransformForUnity.csv", index=False
        )

    print("Mesh → World pipeline completed successfully.")
