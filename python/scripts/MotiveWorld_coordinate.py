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

    # PATHS
    tibia_transform_csv = Path(tibia_transform_csv)
    femur_transform_csv = Path(femur_transform_csv)
    patella_transform_csv = Path(patella_transform_csv)

    tibia_vp_csv = Path(tibia_vp_csv)
    femur_vp_csv = Path(femur_vp_csv)
    patella_vp_csv = Path(patella_vp_csv)

    output_dir = Path(output_dir)
    output_RT_dir = Path(output_RT_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_RT_dir.mkdir(parents=True, exist_ok=True)

    #  LOAD RB → WORLD TRANSFORMS
    tibia_transform = pd.read_csv(tibia_transform_csv)
    femur_transform = pd.read_csv(femur_transform_csv)
    patella_transform = pd.read_csv(patella_transform_csv)

    # qx qy qz qw → scipy [x y z w]
    tibia_rotm = R.from_quat(
        tibia_transform.iloc[:, [1, 2, 3, 4]].to_numpy()
    ).as_matrix()

    femur_rotm = R.from_quat(
        femur_transform.iloc[:, [1, 2, 3, 4]].to_numpy()
    ).as_matrix()

    patella_rotm = R.from_quat(
        patella_transform.iloc[:, [1, 2, 3, 4]].to_numpy()
    ).as_matrix()

    tibia_xyz = tibia_transform.iloc[:, [5, 6, 7]].to_numpy()
    femur_xyz = femur_transform.iloc[:, [5, 6, 7]].to_numpy()
    patella_xyz = patella_transform.iloc[:, [5, 6, 7]].to_numpy()

    assert len(tibia_xyz) == len(femur_xyz) == len(patella_xyz), \
        "Frame mismatch between tibia, femur and patella transforms"

    n_frames = len(tibia_xyz)

    # LOAD RB LOCAL VIRTUAL POINTS
    tibia_vp = pd.read_csv(tibia_vp_csv)
    femur_vp = pd.read_csv(femur_vp_csv)
    patella_vp = pd.read_csv(patella_vp_csv)

    ALT_local = tibia_vp.iloc[0, 0:3].to_numpy().reshape(3, 1)
    AMT_local = tibia_vp.iloc[0, 3:6].to_numpy().reshape(3, 1)
    LT_local  = tibia_vp.iloc[0, 6:9].to_numpy().reshape(3, 1)
    MT_local  = tibia_vp.iloc[0, 9:12].to_numpy().reshape(3, 1)

    AF_local = femur_vp.iloc[0, 0:3].to_numpy().reshape(3, 1)
    LF_local = femur_vp.iloc[0, 3:6].to_numpy().reshape(3, 1)
    MF_local = femur_vp.iloc[0, 6:9].to_numpy().reshape(3, 1)

    DP_local = patella_vp.iloc[0, 0:3].to_numpy().reshape(3, 1)
    MP_local = patella_vp.iloc[0, 3:6].to_numpy().reshape(3, 1)
    LP_local = patella_vp.iloc[0, 6:9].to_numpy().reshape(3, 1)
    PP_local = patella_vp.iloc[0, 9:12].to_numpy().reshape(3, 1)

    # GLOBAL POINT RECONSTRUCTION
    tibia_global = np.zeros((n_frames, 12))
    femur_global = np.zeros((n_frames, 9))
    patella_global = np.zeros((n_frames, 9))

    R_tibia_all = np.zeros((n_frames, 3, 3))
    T_tibia_all = np.zeros((n_frames, 3))

    R_femur_all = np.zeros((n_frames, 3, 3))
    T_femur_all = np.zeros((n_frames, 3))

    R_patella_all = np.zeros((n_frames, 3, 3))
    T_patella_all = np.zeros((n_frames, 3))

    for i in range(n_frames):

        #  TIBIA 
        R_t = tibia_rotm[i]
        T_t = tibia_xyz[i].reshape(3, 1)

        R_tibia_all[i] = R_t
        T_tibia_all[i] = T_t.ravel()

        tibia_global[i, 0:3]  = (R_t @ ALT_local + T_t).ravel()
        tibia_global[i, 3:6]  = (R_t @ AMT_local + T_t).ravel()
        tibia_global[i, 6:9]  = (R_t @ LT_local  + T_t).ravel()
        tibia_global[i, 9:12] = (R_t @ MT_local  + T_t).ravel()

        #  FEMUR 
        R_f = femur_rotm[i]
        T_f = femur_xyz[i].reshape(3, 1)

        R_femur_all[i] = R_f
        T_femur_all[i] = T_f.ravel()

        femur_global[i, 0:3] = (R_f @ AF_local + T_f).ravel()
        femur_global[i, 3:6] = (R_f @ LF_local + T_f).ravel()
        femur_global[i, 6:9] = (R_f @ MF_local + T_f).ravel()

        #  PATELLA 
        R_p = patella_rotm[i]
        T_p = patella_xyz[i].reshape(3, 1)

        R_patella_all[i] = R_p
        T_patella_all[i] = T_p.ravel()

        patella_global[i, 0:3]  = (R_p @ MP_local + T_p).ravel()
        patella_global[i, 3:6]  = (R_p @ LP_local + T_p).ravel()
        patella_global[i, 6:9] = (R_p @ PP_local + T_p).ravel()

    # mm → m
    tibia_global   /= 1000.0
    femur_global   /= 1000.0
    patella_global /= 1000.0

    # SAVE RT
    np.savez(output_RT_dir / "rb_to_world_tibia.npz", R=R_tibia_all, T=T_tibia_all)
    np.savez(output_RT_dir / "rb_to_world_femur.npz", R=R_femur_all, T=T_femur_all)
    np.savez(output_RT_dir / "rb_to_world_patella.npz", R=R_patella_all, T=T_patella_all)

    # WRITE CSV

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
    "patellaMP_X","patellaMP_Y","patellaMP_Z",
    "patellaLP_X","patellaLP_Y","patellaLP_Z",
    "patellaPP_X","patellaPP_Y","patellaPP_Z"
] 

    pd.DataFrame(tibia_global, columns=tibia_cols).to_csv(
    output_dir / "tibiaGlobalPoints.csv", index=False
)

    pd.DataFrame(femur_global, columns=femur_cols).to_csv(
    output_dir / "femurGlobalPoints.csv", index=False
)

    pd.DataFrame(patella_global, columns=patella_cols).to_csv(
    output_dir / "patellaGlobalPoints.csv", index=False
)

