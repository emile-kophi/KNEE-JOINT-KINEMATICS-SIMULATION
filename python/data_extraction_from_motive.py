import pandas as pd
# =========================
# INPUT
# =========================
src = "../files/dataRigidBody.csv"

bodies = {
    "Tibia": "tibiaTransform.csv",
    "Femur": "femurTransform.csv",
}

frame_col = ("Frame", "", "")

# =========================
# LETTURA CSV
# =========================
df = pd.read_csv(
    src,
    sep=";",
    header=[0, 1, 2],
    decimal=",",
    engine="python"
)

# =========================
# NORMALIZZAZIONE HEADER
# =========================
def normalize_tuple(col):
    return tuple("" if x != x else str(x).strip() for x in col)

df.columns = pd.MultiIndex.from_tuples(
    [normalize_tuple(c) for c in df.columns]
)

# =========================
# FLATTEN HEADER
# =========================
def flat_name(col):
    if col == frame_col:
        return "Frame"
    return "_".join(p for p in col if p)

# =========================
# LOOP SU TIBIA / FEMUR
# =========================
for body, file in bodies.items():

    body_cols = [
        (body, "Rotation", "X"),
        (body, "Rotation", "Y"),
        (body, "Rotation", "Z"),
        (body, "Rotation", "W"),
        (body, "Position", "X"),
        (body, "Position", "Y"),
        (body, "Position", "Z"),
    ]

    # Ordine colonne
    var_names = []
    if frame_col in df.columns:
        var_names.append(frame_col)
    var_names += [c for c in body_cols if c in df.columns]

    data_table = df[var_names].copy()

    # Frame fallback
    if frame_col not in df.columns:
        data_table.insert(0, "Frame", range(len(data_table)))

    # Forward fill solo sul rigid body
    valid_cols = [c for c in body_cols if c in data_table.columns]
    data_table[valid_cols] = data_table[valid_cols].ffill()

    # Flatten header
    data_table.columns = [flat_name(c) for c in data_table.columns]

    # Salvataggio
    data_table.to_csv(f"../files/{file}", sep=",", decimal=".", index=False)
    print(f"File creato: {file}")
