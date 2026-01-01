import pandas as pd

def normalize_tuple(col):
    """
    Normalize a MultiIndex column tuple by:
    - converting NaN to empty string
    - stripping whitespace
    """
    return tuple("" if x != x else str(x).strip() for x in col)


def flat_name(col, frame_col):
    """
    Convert a MultiIndex column into a flat string name.
    """
    if col == frame_col:
        return "Frame"
    return "_".join(p for p in col if p)


def Read_dataRB(
    src_csv: str,
    output_dir: str,
    bodies: dict,
    frame_col=("Frame", "", ""),
    sep=";",
    decimal=","
) -> None:
    # READ CSV
    data = pd.read_csv(
        src_csv,
        sep=sep,
        header=[0, 1, 2],
        decimal=decimal,
        engine="python"
    )

    # NORMALIZE MULTIINDEX HEADER
    data.columns = pd.MultiIndex.from_tuples(
        [normalize_tuple(c) for c in data.columns]
    )

    # LOOP OVER RIGID BODIES
    for body, output_file in bodies.items():

        body_cols = [
            (body, "Rotation", "X"),
            (body, "Rotation", "Y"),
            (body, "Rotation", "Z"),
            (body, "Rotation", "W"),
            (body, "Position", "X"),
            (body, "Position", "Y"),
            (body, "Position", "Z"),
        ]

        # Column ordering
        var_names = []
        if frame_col in data.columns:
            var_names.append(frame_col)

        var_names += [c for c in body_cols if c in data.columns]

        data_table = data[var_names].copy()

        # Frame fallback if missing
        if frame_col not in data.columns:
            data_table.insert(0, "Frame", range(len(data_table)))

        # Forward fill only rigid-body columns
        valid_cols = [c for c in body_cols if c in data_table.columns]
        data_table[valid_cols] = data_table[valid_cols].ffill()

        # Flatten header
        data_table.columns = [
            flat_name(c, frame_col) for c in data_table.columns
        ]

        # SAVE CSV
        out_path = f"{output_dir}/{output_file}"
        data_table.to_csv(out_path, sep=",", decimal=".", index=False)

        print(f"File created: {out_path}")
