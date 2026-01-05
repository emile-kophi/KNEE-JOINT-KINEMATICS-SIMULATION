import numpy as np
import pandas as pd
from pathlib import Path

def load_mesh_to_world(mesh_to_world: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load (R, T) from a .npz file produced by Mesh_to_World_Pipeline.
    """
    mesh_to_world = Path(mesh_to_world)
    data = np.load(mesh_to_world)

    R_all = data["R"]
    T_all = data["T"]

    if R_all.ndim != 3 or R_all.shape[1:] != (3, 3):
        raise ValueError(f"Invalid R shape: {R_all.shape}")

    if T_all.ndim != 2 or T_all.shape[1] != 3:
        raise ValueError(f"Invalid T shape: {T_all.shape}")

    return R_all, T_all

def knee_flexion_angle(
    R_world_tibia: np.ndarray,
    T_world_tibia: np.ndarray,
    R_world_femur: np.ndarray,
    T_world_femur: np.ndarray,
    plot: bool = False
) -> np.ndarray:
    """
    Relative knee flexion angle (tibia relative to femur),
    zero-referenced to the first frame.
    """

    n_frames = R_world_tibia.shape[0]

    if R_world_femur.shape[0] != n_frames:
        raise ValueError("Tibia and Femur transforms must have the same number of frames.")

    Rot_B2M = np.diag([-1.0, 1.0, -1.0])

    # Anatomical points (mesh local)
    Tibia_Prox = Rot_B2M @ np.array([-0.001619,  0.186731, -0.012323], dtype=float)
    Tibia_Dist = Rot_B2M @ np.array([ 0.002285, -0.18949,   0.004076], dtype=float)

    Femur_Head = Rot_B2M @ np.array([-0.017593,  0.225445,  0.028621], dtype=float)
    Femur_Dist = Rot_B2M @ np.array([-0.012257, -0.222853, -0.01539 ], dtype=float)

    LF_local = Rot_B2M @ np.array([ 0.022548, -0.207434, -0.032849], dtype=float)
    MF_local = Rot_B2M @ np.array([-0.054633, -0.204439, -0.003766], dtype=float)

    theta_deg = np.zeros(n_frames, dtype=float)

    for i in range(n_frames):

        R_t = R_world_tibia[i]
        T_t = T_world_tibia[i]

        R_f = R_world_femur[i]
        T_f = T_world_femur[i]

        # Femur condylar axis 
        LF = R_f @ LF_local + T_f
        MF = R_f @ MF_local + T_f

        condil_axis = LF - MF
        condil_axis /= np.linalg.norm(condil_axis)

        # Femur mechanical axis (reference)
        Fp = R_f @ Femur_Head + T_f
        Fd = R_f @ Femur_Dist + T_f

        axis_femur = Fd - Fp
        axis_femur /= np.linalg.norm(axis_femur)

        # Reference direction in sagittal plane
        x_ref = axis_femur - np.dot(axis_femur, condil_axis) * condil_axis
        x_ref /= np.linalg.norm(x_ref)

        y_ref = np.cross(condil_axis, x_ref)
        y_ref /= np.linalg.norm(y_ref)

        # Tibia mechanical axis
        Pp = R_t @ Tibia_Prox + T_t
        Pd = R_t @ Tibia_Dist + T_t

        axis_tibia = Pd - Pp
        axis_tibia /= np.linalg.norm(axis_tibia)

        tibia_proj = axis_tibia - np.dot(axis_tibia, condil_axis) * condil_axis
        tibia_proj /= np.linalg.norm(tibia_proj)

        # relative angle 
        cosang = np.clip(np.dot(x_ref, tibia_proj), -1.0, 1.0)
        sinang = np.dot(y_ref, tibia_proj)

        theta_deg[i] = np.degrees(np.arctan2(sinang, cosang))

    # ZERO-REFERENCING (standard in biomechanics)
    theta_base=theta_deg[0]
    theta_deg = theta_deg - theta_deg[0]
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(theta_deg)
        plt.xlabel(f"Frame | Reference angle {theta_base:.0f}° ")
        plt.ylabel("Flexion angle (deg)")
        plt.grid(True)
        plt.show()

    return theta_deg


# RMS RECONSTRUCTION ERROR (MOTIVE WORLD)
def RMS_error(
    R_world_tibia: np.ndarray,
    T_world_tibia: np.ndarray,
    R_world_femur: np.ndarray,
    T_world_femur: np.ndarray,
    R_world_patella: np.ndarray,
    T_world_patella: np.ndarray,
    meshtibia_local: np.ndarray,
    meshfemur_local: np.ndarray,
    meshpatella_local: np.ndarray,
    tibia_global_csv: str | Path,
    femur_global_csv: str | Path,
    patella_global_csv: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    RMS reconstruction error between:
    reconstructed mesh points (mesh → world)
    and Motive global marker points.

    NOTE:
    - R_world, T_world are Unity world (LH)
    - GlobalPoints.csv are Motive world (RH)
    """

    # Unity ↔ Motive conversion (self-inverse)
    S = np.diag([1.0, 1.0, -1.0])

    T_t = pd.read_csv(tibia_global_csv)
    T_f = pd.read_csv(femur_global_csv)
    T_p = pd.read_csv(patella_global_csv)

    def getB(T, i, p):
        return np.array([
            T.loc[i, f"{p}_X"],
            T.loc[i, f"{p}_Y"],
            T.loc[i, f"{p}_Z"],
        ], dtype=float)

    n_frames = R_world_tibia.shape[0]

    err_tibia   = np.zeros(n_frames)
    err_femur   = np.zeros(n_frames)
    err_patella = np.zeros(n_frames)

    for i in range(n_frames):

        # TIBIA 
        P_rec_unity = (
            R_world_tibia[i] @ (S @ meshtibia_local)
            + T_world_tibia[i][:, None]
        )
        P_rec = S @ P_rec_unity   # back to Motive world

        P_true = np.column_stack([
            getB(T_t, i, "tibiaAMT"),
            getB(T_t, i, "tibiaLT"),
            getB(T_t, i, "tibiaMT"),
        ])

        err_tibia[i] = np.sqrt(np.mean(np.sum((P_rec - P_true) ** 2, axis=0)))

        # FEMUR
        P_rec_unity = (
            R_world_femur[i] @ (S @ meshfemur_local)
            + T_world_femur[i][:, None]
        )
        P_rec = S @ P_rec_unity

        P_true = np.column_stack([
            getB(T_f, i, "femurAF"),
            getB(T_f, i, "femurLF"),
            getB(T_f, i, "femurMF"),
        ])

        err_femur[i] = np.sqrt(np.mean(np.sum((P_rec - P_true) ** 2, axis=0)))

        # PATELLA
        P_rec_unity = (
            R_world_patella[i] @ (S @ meshpatella_local)
            + T_world_patella[i][:, None]
        )
        P_rec = S @ P_rec_unity

        P_true = np.column_stack([
            getB(T_p, i, "patellaPP"),
            getB(T_p, i, "patellaMP"),
            getB(T_p, i, "patellaLP"),
        ])

        err_patella[i] = np.sqrt(np.mean(np.sum((P_rec - P_true) ** 2, axis=0)))

    return err_tibia, err_femur, err_patella
