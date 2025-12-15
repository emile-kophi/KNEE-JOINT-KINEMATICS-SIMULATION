import pandas as pd
from scipy.io import loadmat
from pathlib import Path


def export_virtual_points(
    mat_file: str | Path,
    output_dir: str | Path,
    virtual_points: dict
) -> None:
    """
    Export virtual points stored in a MATLAB .mat file to CSV format.
    """

    mat_file = Path(mat_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # LOAD .MAT FILE
    mat_data = loadmat(mat_file)

    # LOOP OVER BODIES
    for body, cfg in virtual_points.items():

        data_dict = {}

        for point_name in cfg["points"]:

            if point_name in mat_data:
                coords = mat_data[point_name].flatten()
                x, y, z = coords[0], coords[1], coords[2]
            else:
                print(
                    f"Warning: {point_name} not found in {mat_file}. "
                    "Using [0, 0, 0]."
                )
                x, y, z = 0.0, 0.0, 0.0

            data_dict[f"{point_name}_X"] = [x]
            data_dict[f"{point_name}_Y"] = [y]
            data_dict[f"{point_name}_Z"] = [z]

        # DATAFRAME + SAVE
        out_path = output_dir / cfg["output"]
        data_table = pd.DataFrame(data_dict)
        data_table.to_csv(out_path, index=False, sep=",")

        print(f"File created: {out_path}")
