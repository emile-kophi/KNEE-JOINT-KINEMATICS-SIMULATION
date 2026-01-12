import numpy as np
import pandas as pd
from pathlib import Path
import numpy as np

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Zero-norm vector.")
    return v / n


def load_mesh_to_world(path: str):
    """
    Load (R_all, T_all) from a .npz file.

    Expected:
      R: (n, 3, 3)
      T: (n, 3)
    """
    data = np.load(path)
    R_all = data["R"]
    T_all = data["T"]

    if R_all.ndim != 3 or R_all.shape[1:] != (3, 3):
        raise ValueError(f"Invalid R shape: {R_all.shape}")
    if T_all.ndim != 2 or T_all.shape[1] != 3:
        raise ValueError(f"Invalid T shape: {T_all.shape}")

    return R_all, T_all

# FEMUR FRAME
def femur_frame_world(
    HF_w: np.ndarray,
    LF_w: np.ndarray,
    MF_w: np.ndarray,
    *,
    side: str
) -> np.ndarray:
    """
    Femur anatomical frame (Grood & Suntay, MATLAB-equivalent).
    """

    Of = 0.5 * (LF_w + MF_w)

    if side.upper() == "RIGHT":
        zF = normalize(Of - HF_w)
    else:
        zF = normalize(HF_w - Of)

    yF = normalize(np.cross(zF, (LF_w - MF_w)))
    xF = normalize(np.cross(yF, zF))

    return np.column_stack((xF, yF, zF))

# TIBIA FRAME
def tibia_frame_world(
    LT_w: np.ndarray,
    MT_w: np.ndarray,
    AMT_w: np.ndarray,
    ALT_w: np.ndarray,
    *,
    side: str
) -> np.ndarray:
    """
    Tibia anatomical frame (Grood & Suntay, MATLAB-equivalent).
    """

    Ot = 0.5 * (LT_w + MT_w)
    AT = 0.45 * ALT_w + 0.55 * AMT_w

    if side.upper() == "RIGHT":
        zT = normalize(AT - Ot)
    else:
        zT = normalize(Ot - AT)

    yT = normalize(np.cross(zT, (LT_w - MT_w)))
    xT = normalize(np.cross(yT, zT))

    return np.column_stack((xT, yT, zT))


# GROOD & SUNTAY ANGLES
def grood_suntay_angles(side: str, FemToTib: np.ndarray):
    """
    Compute FE, VV, IE from Femur→Tibia rotation matrix.
    """

    # Femur reference axes
    e1  = np.array([1.0, 0.0, 0.0])  # ML
    e1r = np.array([0.0, 1.0, 0.0])  # AP

    # Tibia axes in femur frame
    e3 = FemToTib[:, 2]   # tibia z
    i  = FemToTib[:, 0]   # tibia x

    # Floating axis
    e2 = normalize(np.cross(e3, e1))

    # Flexion / Extension
    p = np.dot(np.cross(e1r, e2), e1)
    n = 1.0 if abs(p) < 1e-12 else np.sign(p)

    FE = np.arctan2(
        n * np.linalg.norm(np.cross(e1r, e2)),
        np.dot(e1r, e2)
    )

    # Varus / Valgus + Internal / External
    c = np.clip(np.dot(e1, e3), -1.0, 1.0)

    if side.upper() == "LEFT":
        VV = -(np.pi / 2.0 - np.arccos(c))
        IE = np.arcsin(np.clip(np.dot(e2, i), -1.0, 1.0))
    else:
        VV = (np.arccos(c) - np.pi / 2.0)
        IE = -np.arcsin(np.clip(np.dot(-e2, i), -1.0, 1.0))

    return np.degrees(FE), np.degrees(VV), np.degrees(IE)

# FULL DYNAMIC PIPELINE
def grood_suntay_pipeline(
    R_femur: np.ndarray,
    T_femur: np.ndarray,
    R_tibia: np.ndarray,
    T_tibia: np.ndarray,
    anatomy: dict,
    *,
    side: str = "LEFT"
):
    """
    Compute FE, VV, IE using MATLAB-equivalent anatomical frames.

    Inputs:
      R_*, T_* : mesh→world transforms (WORLD MOTIVE)
      anatomy  : mesh-local anatomical points
    """

    n = min(len(R_femur), len(R_tibia))

    FE = np.zeros(n)
    VV = np.zeros(n)
    IE = np.zeros(n)

    # Mesh-local landmarks
    HF_l = anatomy["femur"]["head"]
    LF_l = anatomy["femur"]["LF"]
    MF_l = anatomy["femur"]["MF"]

    LT_l  = anatomy["tibia"]["LT"]
    MT_l  = anatomy["tibia"]["MT"]
    AMT_l = anatomy["tibia"]["AMT"]
    ALT_l = anatomy["tibia"]["ALT"]

    for k in range(n):

        # FEMUR landmarks → WORLD
        HF_w = R_femur[k] @ HF_l + T_femur[k]
        LF_w = R_femur[k] @ LF_l + T_femur[k]
        MF_w = R_femur[k] @ MF_l + T_femur[k]

        # TIBIA landmarks → WORLD
        LT_w  = R_tibia[k] @ LT_l  + T_tibia[k]
        MT_w  = R_tibia[k] @ MT_l  + T_tibia[k]
        AMT_w = R_tibia[k] @ AMT_l + T_tibia[k]
        ALT_w = R_tibia[k] @ ALT_l + T_tibia[k]

        # Anatomical frames
        Rf = femur_frame_world(HF_w, LF_w, MF_w, side=side)
        Rt = tibia_frame_world(LT_w, MT_w, AMT_w, ALT_w, side=side)

        # Relative rotation
        FemToTib = Rf.T @ Rt

        # Angles
        FE[k], VV[k], IE[k] = grood_suntay_angles(side, FemToTib)

    return FE, VV, IE


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
        anatomy["tibia"]["MT"]
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
        P_rec= (
            R_world_tibia[i] @ (P_tibia_local)
            + T_world_tibia[i][:, None]
        )


        P_true = np.column_stack([
            getB(T_t, i, "tibiaAMT"),
            getB(T_t, i, "tibiaLT"),
            getB(T_t, i, "tibiaMT"),
        ])

        err_tibia[i] = np.sqrt(np.mean(np.sum((P_rec - P_true) ** 2, axis=0)))

        # FEMUR 
        P_rec= (
            R_world_femur[i] @ (P_femur_local)
            + T_world_femur[i][:, None]
        )

        P_true = np.column_stack([
            getB(T_f, i, "femurAF"),
            getB(T_f, i, "femurLF"),
            getB(T_f, i, "femurMF"),
        ])

        err_femur[i] = np.sqrt(np.mean(np.sum((P_rec - P_true) ** 2, axis=0)))

        # PATELLA 
        P_rec= (
            R_world_patella[i] @ (P_patella_local)
            + T_world_patella[i][:, None]
        )

        P_true = np.column_stack([
            getB(T_p, i, "patellaPP"),
            getB(T_p, i, "patellaMP"),
            getB(T_p, i, "patellaLP"),
        ])

        err_patella[i] = np.sqrt(np.mean(np.sum((P_rec - P_true) ** 2, axis=0)))

    return err_tibia, err_femur, err_patella


