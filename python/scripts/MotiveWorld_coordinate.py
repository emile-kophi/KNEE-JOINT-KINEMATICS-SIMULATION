import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from pathlib import Path


def Global_points(
    tibia_transform_csv: str | Path,
    femur_transform_csv: str | Path,
    patella_transform_csv: str | Path,
    tibia_vp_csv: str | Path,
    femur_vp_csv: str | Path,
    patella_vp_csv: str | Path,
    output_dir: str | Path,
    output_RT_dir: str | Path,
) -> None:
    """
    Generate global marker points for tibia and femur using:
    - RB → world transforms (from Motive)
    - local RB virtual points
    """
    output_RT_dir = Path(output_RT_dir)

    tibia_transform_csv = Path(tibia_transform_csv)
    femur_transform_csv = Path(femur_transform_csv)
    patella_transform_csv = Path(patella_transform_csv)
    tibia_vp_csv = Path(tibia_vp_csv)
    femur_vp_csv = Path(femur_vp_csv)
    patella_vp_csv = Path(patella_vp_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) LOAD RB → WORLD TRANSFORMS (Motive)
    tibia_transform = pd.read_csv(tibia_transform_csv)
    femur_transform = pd.read_csv(femur_transform_csv)
    patella_transform = pd.read_csv(patella_transform_csv)

    # CSV order: qx qy qz qw → MATLAB order: [qw qx qy qz]
    tibia_q = tibia_transform.iloc[:, [4, 1, 2, 3]].to_numpy()
    femur_q = femur_transform.iloc[:, [4, 1, 2, 3]].to_numpy()
    patella_q = patella_transform.iloc[:, [4, 1, 2, 3]].to_numpy()

    tibia_rotm = R.from_quat(
        np.column_stack([
            tibia_q[:, 1],
            tibia_q[:, 2],
            tibia_q[:, 3],
            tibia_q[:, 0]
        ])
    ).as_matrix()

    femur_rotm = R.from_quat(
        np.column_stack([
            femur_q[:, 1],
            femur_q[:, 2],
            femur_q[:, 3],
            femur_q[:, 0]
        ])
    ).as_matrix()
    patella_rotm = R.from_quat(
        np.column_stack([
            patella_q[:, 1],
            patella_q[:, 2],
            patella_q[:, 3],
            patella_q[:, 0]
        ])
    ).as_matrix()

    tibia_xyz = tibia_transform.iloc[:, [5, 6, 7]].to_numpy()
    femur_xyz = femur_transform.iloc[:, [5, 6, 7]].to_numpy()
    patella_xyz = patella_transform.iloc[:, [5, 6, 7]].to_numpy()

    # 2) LOAD LOCAL RB POINTS (virtual points)
    tibia_vp = pd.read_csv(tibia_vp_csv)
    femur_vp = pd.read_csv(femur_vp_csv)
    patella_vp = pd.read_csv(patella_vp_csv)

    # Tibia locals
    ALT_local = tibia_vp.iloc[0, [0, 1, 2]].to_numpy().reshape(3, 1)
    AMT_local = tibia_vp.iloc[0, [3, 4, 5]].to_numpy().reshape(3, 1)
    LT_local  = tibia_vp.iloc[0, [6, 7, 8]].to_numpy().reshape(3, 1)
    MT_local  = tibia_vp.iloc[0, [9, 10, 11]].to_numpy().reshape(3, 1)

    # Femur locals
    AF_local = femur_vp.iloc[0, [0, 1, 2]].to_numpy().reshape(3, 1)
    LF_local = femur_vp.iloc[0, [3, 4, 5]].to_numpy().reshape(3, 1)
    MF_local = femur_vp.iloc[0, [6, 7, 8]].to_numpy().reshape(3, 1)

     # Patella locals
    DP_local = patella_vp.iloc[0, [0, 1, 2]].to_numpy().reshape(3, 1)
    MP_local = patella_vp.iloc[0, [3, 4, 5]].to_numpy().reshape(3, 1)
    LP_local = patella_vp.iloc[0, [6, 7, 8]].to_numpy().reshape(3, 1)
    PP_local = patella_vp.iloc[0, [9, 10, 11]].to_numpy().reshape(3, 1)

    # 3) GLOBAL POINT RECONSTRUCTION
    n_frames = tibia_xyz.shape[0]

    tibia_global = np.zeros((n_frames, 12))
    femur_global = np.zeros((n_frames, 9))
    patella_global = np.zeros((n_frames, 12))

    for i in range(n_frames):

        R_t = tibia_rotm[i]
        T_t = tibia_xyz[i].reshape(3, 1)

        tibia_global[i, 0:3]  = (R_t @ ALT_local + T_t).ravel()
        tibia_global[i, 3:6]  = (R_t @ AMT_local + T_t).ravel()
        tibia_global[i, 6:9]  = (R_t @ LT_local  + T_t).ravel()
        tibia_global[i, 9:12] = (R_t @ MT_local  + T_t).ravel()

        R_f = femur_rotm[i]
        T_f = femur_xyz[i].reshape(3, 1)

        femur_global[i, 0:3] = (R_f @ AF_local + T_f).ravel()
        femur_global[i, 3:6] = (R_f @ LF_local + T_f).ravel()
        femur_global[i, 6:9] = (R_f @ MF_local + T_f).ravel()

        R_p = patella_rotm[i]
        T_p = patella_xyz[i].reshape(3, 1)

        patella_global[i, 0:3]  = (R_p @ DP_local + T_p).ravel()
        patella_global[i, 3:6]  = (R_p @ MP_local + T_p).ravel()
        patella_global[i, 6:9]  = (R_p @ LP_local  + T_p).ravel()
        patella_global[i, 9:12] = (R_p @ PP_local  + T_p).ravel()        



    # MATLAB: mm → m
    tibia_global /= 1000
    femur_global /= 1000
    patella_global /= 1000
    np.savez(output_RT_dir / "rb_to_world_tibia.npz", R=R_t, T=T_t)
    np.savez(output_RT_dir / "rb_to_wrld_femur.npz", R=R_f, T=T_f)
    np.savez(output_RT_dir / "rb_to_wrld_patella.npz", R=R_p, T=T_p)

    # 4) WRITE CSV
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
    patella_cols = [
        "patellaDP_X","patellaDP_Y","patellaDP_Z",
        "patellaMP_X","patellaMP_Y","patellaMP_Z",
        "patellaLP_X","patellaLP_Y","patellaLP_Z",
        "patellaPP_X","patellaPP_Y","patellaPP_Z"
    ]


    pd.DataFrame(
        tibia_global, columns=tibia_cols
    ).to_csv(output_dir / "tibiaGlobalPoints.csv", index=False)

    pd.DataFrame(
        femur_global, columns=femur_cols
    ).to_csv(output_dir / "femurGlobalPoints.csv", index=False)
    pd.DataFrame(
        patella_global, columns=patella_cols
    ).to_csv(output_dir / "patellaGlobalPoints.csv", index=False)

    print("Global points generated successfully.")
