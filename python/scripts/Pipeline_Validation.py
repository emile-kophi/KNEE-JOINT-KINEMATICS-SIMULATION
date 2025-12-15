import numpy as np
import pandas as pd
from pathlib import Path

# helpers

def load_mesh_to_world(mesh_to_world: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load (R, T) from a .npz file produced by Mesh_to_World_Pipeline.
    """
    mesh_to_world = Path(mesh_to_world)
    data = np.load(mesh_to_world)
    R_all = data["R"]
    T_all = data["T"]

    if R_all.ndim != 3 or R_all.shape[1:] != (3, 3):
        raise ValueError(f"Invalid R shape in {mesh_to_world}: {R_all.shape}. Expected (N,3,3).")

    if T_all.ndim != 2 or T_all.shape[1] != 3:
        raise ValueError(f"Invalid T shape in {mesh_to_world}: {T_all.shape}. Expected (N,3).")

    return R_all, T_all

#  Knee flexion angle (pure, no quaternion recomputation)

def knee_flexion_angle(
    R_world_tibia: np.ndarray,   # (N,3,3)
    T_world_tibia: np.ndarray,   # (N,3)
    R_world_femur: np.ndarray,   # (N,3,3)
    T_world_femur: np.ndarray,   # (N,3)
    plot: bool = False
) -> np.ndarray:
    """
    Compute anatomical knee flexion angle over time using ONLY the
    mesh->world transforms already computed by the pipeline.
    """

    n_frames = R_world_tibia.shape[0]

    if R_world_femur.shape[0] != n_frames:
        raise ValueError("Tibia and Femur transforms must have the same number of frames.")

    # Unity → Motive-like correction used in your pipeline (Blender-like)
    C = np.diag([-1.0, 1.0, -1.0])

    # Anatomical points (Unity local)
    Tibia_Prox = C @ np.array([0.0008561252, 0.09804206, 0.01295095], dtype=float)
    Tibia_Dist = C @ np.array([0.003904605,  0.3762208,  0.01639927], dtype=float)

    Femur_Prox = C @ np.array([0.03055387,  -0.3575693,  0.0149763], dtype=float)
    Femur_Dist = C @ np.array([0.004923976, -0.04483421, 0.01108846], dtype=float)

    # Femur condyles axis (Unity local)
    LF_local = C @ np.array([ 0.03755924, -0.001844588, -0.01695582], dtype=float)
    MF_local = C @ np.array([-0.03962225, -0.004840371,  0.01212757], dtype=float)

    theta_deg = np.zeros(n_frames, dtype=float)

    for i in range(n_frames):

        # Tibia anatomical axis in world
        R_t = R_world_tibia[i]
        T_t = T_world_tibia[i]

        Pp = R_t @ Tibia_Prox + T_t
        Pd = R_t @ Tibia_Dist + T_t

        axis_tibia = Pd - Pp
        axis_tibia /= np.linalg.norm(axis_tibia)

        # Femur anatomical axis in world
        R_f = R_world_femur[i]
        T_f = T_world_femur[i]

        Fp = R_f @ Femur_Prox + T_f
        Fd = R_f @ Femur_Dist + T_f

        axis_femur = Fd - Fp
        axis_femur /= np.linalg.norm(axis_femur)

        # Condyle axis in world (LF - MF)
        LF = R_f @ LF_local + T_f
        MF = R_f @ MF_local + T_f

        condyle_axis = LF - MF
        condyle_axis /= np.linalg.norm(condyle_axis)

        # Project tibia axis onto femur sagittal plane
        t_proj = axis_tibia - np.dot(axis_tibia, condyle_axis) * condyle_axis
        t_proj /= np.linalg.norm(t_proj)

        # Flexion angle
        cosang = np.clip(np.dot(axis_femur, t_proj), -1.0, 1.0)
        theta_deg[i] = np.degrees(np.arccos(cosang))

    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(theta_deg)
        plt.xlabel("Frame")
        plt.ylabel("Flexion angle (deg)")
        plt.grid(True)
        plt.show()

    return theta_deg


# Reconstruction RMS error

def RMS_error(
    R_world_tibia: np.ndarray,     # (N,3,3)
    T_world_tibia: np.ndarray,     # (N,3)
    R_world_femur: np.ndarray,     # (N,3,3)
    T_world_femur: np.ndarray,     # (N,3)
    meshtibia_local: np.ndarray,   # (3,3)
    meshfemur_local: np.ndarray,   # (3,3)
    tibia_global_csv: str | Path,
    femur_global_csv: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute RMS reconstruction error between:
    - reconstructed mesh points (mesh → world)
    - true Motive global marker points
    """

    tibia_global_csv = Path(tibia_global_csv)
    femur_global_csv = Path(femur_global_csv)

    T_t = pd.read_csv(tibia_global_csv)
    T_f = pd.read_csv(femur_global_csv)

    def getB(T, i, p):
        return np.array([
            T.loc[i, f"{p}_X"],
            T.loc[i, f"{p}_Y"],
            T.loc[i, f"{p}_Z"],
        ], dtype=float)

    n_frames = R_world_tibia.shape[0]

    if R_world_femur.shape[0] != n_frames:
        raise ValueError("Tibia and Femur transforms must have the same number of frames.")

    err_tibia = np.zeros(n_frames, dtype=float)
    err_femur = np.zeros(n_frames, dtype=float)

    for i in range(n_frames):

        # TIBIA
        P_rec = (
            R_world_tibia[i] @ meshtibia_local
            + T_world_tibia[i][:, None]
        )

        P_true = np.column_stack([
            getB(T_t, i, "tibiaAMT"),
            getB(T_t, i, "tibiaLT"),
            getB(T_t, i, "tibiaMT"),
        ])

        err_tibia[i] = np.sqrt(np.mean(np.sum((P_rec - P_true) ** 2, axis=0)))

        # FEMUR 
        P_rec = (
            R_world_femur[i] @ meshfemur_local
            + T_world_femur[i][:, None]
        )

        P_true = np.column_stack([
            getB(T_f, i, "femurAF"),
            getB(T_f, i, "femurLF"),
            getB(T_f, i, "femurMF"),
        ])

        err_femur[i] = np.sqrt(np.mean(np.sum((P_rec - P_true) ** 2, axis=0)))

    return err_tibia, err_femur
