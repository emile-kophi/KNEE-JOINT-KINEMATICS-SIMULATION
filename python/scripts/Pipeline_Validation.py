import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

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
    anatomy: dict,
    plot: bool = False
) -> np.ndarray:
    """
    Relative knee flexion angle (tibia relative to femur),
    zero-referenced to the first frame.
    """

    n_frames = R_world_tibia.shape[0]

    if R_world_femur.shape[0] != n_frames:
        raise ValueError("Tibia and Femur transforms must have the same number of frames.")

    # Anatomical points (mesh local)
    Tibia_Prox = anatomy["tibia"]["prox"]
    Tibia_Dist = anatomy["tibia"]["dist"]

    Femur_Head = anatomy["femur"]["head"]
    Femur_Dist = anatomy["femur"]["dist"]

    LF_local = anatomy["femur"]["LF"]
    MF_local = anatomy["femur"]["MF"]

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
        plt.figure()
        plt.plot(theta_deg)
        plt.xlabel(f"Frame | Reference angle {theta_base:.0f}° ")
        plt.ylabel("Flexion angle (deg)")
        plt.grid(True)
        plt.show()

    return theta_deg

import numpy as np

def knee_varus_valgus_angle(
    R_world_tibia: np.ndarray,
    T_world_tibia: np.ndarray,
    R_world_femur: np.ndarray,
    T_world_femur: np.ndarray,
    anatomy: dict,
    plot: bool = False
) -> np.ndarray:
    """
    Relative knee varus–valgus angle (tibia relative to femur),
    computed as the component of the relative rotation about the femoral
    AP axis (normal to the femoral frontal plane), zero-referenced to frame 0.

    Notes:
    - Robust for non-planar / real motions (rotation is generally oblique).
    - Sign convention depends on the orientation of the femoral AP axis,
      which depends on (LF, MF) ordering and the definition of the femoral
      mechanical axis.
    """

    n_frames = R_world_tibia.shape[0]
    if R_world_femur.shape[0] != n_frames:
        raise ValueError("Tibia and Femur transforms must have the same number of frames.")

    # --- Local anatomical points (mesh local) ---
    Femur_AF   = anatomy["femur"]["head"]
    Femur_Dist = anatomy["femur"]["dist"]
    LF_local   = anatomy["femur"]["LF"]
    MF_local   = anatomy["femur"]["MF"]

    alpha_deg = np.zeros(n_frames, dtype=float)

    # Temporal continuity of AP axis (optional but robust)
    x_prev = None

    for i in range(n_frames):

        R_t = R_world_tibia[i]
        R_f = R_world_femur[i]
        T_f = T_world_femur[i]

        # --------------------------------------------------
        # Femur mediolateral axis (condylar axis) in world
        # --------------------------------------------------
        LF = R_f @ LF_local + T_f
        MF = R_f @ MF_local + T_f

        ml = LF - MF
        ml_norm = np.linalg.norm(ml)
        if ml_norm < 1e-12:
            raise ValueError(f"Degenerate ML axis at frame {i}.")
        ml = ml / ml_norm

        # --------------------------------------------------
        # Femur mechanical axis (longitudinal) in world
        # --------------------------------------------------
        Fp = R_f @ Femur_AF   + T_f
        Fd = R_f @ Femur_Dist + T_f

        lf = Fd - Fp
        lf_norm = np.linalg.norm(lf)
        if lf_norm < 1e-12:
            raise ValueError(f"Degenerate femur axis at frame {i}.")
        lf = lf / lf_norm

        # --------------------------------------------------
        # Femur AP axis (normal to frontal plane) in world
        # --------------------------------------------------
        x = np.cross(ml, lf)
        x_norm = np.linalg.norm(x)
        if x_norm < 1e-12:
            raise ValueError(f"Degenerate AP axis at frame {i} (ml and lf nearly collinear).")
        x = x / x_norm

        # Enforce temporal continuity of x (avoid sign flips)
        if x_prev is not None:
            if np.dot(x, x_prev) < 0.0:
                x = -x
        x_prev = x.copy()

        # --------------------------------------------------
        # Relative rotation femur->tibia
        # --------------------------------------------------
        R_rel = R_f.T @ R_t  # tibia relative to femur (SO(3))

        # Log map SO(3) -> rotation vector (axis * angle), in femur local coords
        w_femur = _so3_log(R_rel)  # radians

        # Convert to world for projection onto world-axis x
        w_world = R_f @ w_femur

        # Varus–valgus component: projection onto femoral AP axis
        alpha_deg[i] = np.degrees(np.dot(w_world, x))

    # Zero-referencing
    alpha_base=alpha_deg[0]
    alpha_deg = alpha_deg - alpha_deg[0]

    # Optional plotting
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(alpha_deg)
        plt.xlabel(f"Frame | {alpha_base:.2f}°")
        plt.ylabel("Varus–Valgus angle (deg)")
        plt.grid(True)
        plt.show()

    return alpha_deg


def _so3_log(R: np.ndarray) -> np.ndarray:
    """
    Log map from SO(3) to so(3) rotation vector (axis * angle), in radians.
    Returns a 3-vector w such that exp([w]x) = R.
    """
    tr = float(np.trace(R))
    cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))

    if theta < 1e-12:
        return np.zeros(3, dtype=float)

    # If theta is close to pi, numerical issues can appear; handle robustly
    if np.pi - theta < 1e-6:
        # Use diagonal-based axis extraction
        A = (R + np.eye(3)) / 2.0
        axis = np.array([np.sqrt(max(A[0, 0], 0.0)),
                         np.sqrt(max(A[1, 1], 0.0)),
                         np.sqrt(max(A[2, 2], 0.0))], dtype=float)

        # Fix signs using off-diagonals
        if R[2, 1] - R[1, 2] < 0:
            axis[0] = -axis[0]
        if R[0, 2] - R[2, 0] < 0:
            axis[1] = -axis[1]
        if R[1, 0] - R[0, 1] < 0:
            axis[2] = -axis[2]

        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-12:
            return np.zeros(3, dtype=float)

        axis = axis / axis_norm
        return axis * theta

    # Standard case
    S = (R - R.T) / (2.0 * np.sin(theta))
    w = np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=float) * theta
    return w




def RMS_error(
    R_world_tibia: np.ndarray,
    T_world_tibia: np.ndarray,
    R_world_femur: np.ndarray,
    T_world_femur: np.ndarray,
    R_world_patella: np.ndarray,
    T_world_patella: np.ndarray,
    anatomy: dict,
    tibia_global_csv: str | Path,
    femur_global_csv: str | Path,
    patella_global_csv: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    RMS reconstruction error between reconstructed mesh points (mesh → world)
    and Motive global marker points.
    Assumptions:
    - R_world*, T_world* are in Unity world (LH)
    - GlobalPoints.csv are in Motive world (RH)
    """

    # Unity ↔ Motive conversion (self-inverse)
    S = np.diag([1.0, 1.0, -1.0])

    # Load Motive global points
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

    err_tibia   = np.zeros(n_frames, dtype=float)
    err_femur   = np.zeros(n_frames, dtype=float)
    err_patella = np.zeros(n_frames, dtype=float)

    # Build LOCAL mesh point matrices ONCE (3 × N)
    P_tibia_local = np.column_stack([
        anatomy["tibia"]["AMT"],
        anatomy["tibia"]["LT"],
        anatomy["tibia"]["MT"],
    ])

    P_femur_local = np.column_stack([
        anatomy["femur"]["head"],
        anatomy["femur"]["LF"],
        anatomy["femur"]["MF"],
    ])

    P_patella_local = np.column_stack([
        anatomy["patella"]["PP"],
        anatomy["patella"]["MP"],
        anatomy["patella"]["LP"],
    ])

    # Frame loop
    for i in range(n_frames):

        # TIBIA 
        P_rec_unity = (
            R_world_tibia[i] @ (S @ P_tibia_local)
            + T_world_tibia[i][:, None]
        )
        P_rec = S @ P_rec_unity  # back to Motive world

        P_true = np.column_stack([
            getB(T_t, i, "tibiaAMT"),
            getB(T_t, i, "tibiaLT"),
            getB(T_t, i, "tibiaMT"),
        ])

        err_tibia[i] = np.sqrt(np.mean(np.sum((P_rec - P_true) ** 2, axis=0)))

        # FEMUR 
        P_rec_unity = (
            R_world_femur[i] @ (S @ P_femur_local)
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
            R_world_patella[i] @ (S @ P_patella_local)
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


