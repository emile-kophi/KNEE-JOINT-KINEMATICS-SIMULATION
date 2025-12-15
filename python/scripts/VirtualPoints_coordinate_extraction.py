import pandas as pd
from scipy.io import loadmat
from pathlib import Path

# PATHS
BASE_DIR = Path(__file__).resolve().parent      # virtual/python
FILES_DIR = BASE_DIR.parent / "../data/matlab"
SAVE_DIR = BASE_DIR.parent / "../data/csv_files"  # virtual/files

mat_file = FILES_DIR / "virtualPoint_DEMO.mat"

virtual_points = {
    "Femur": {
        "points": ["femurAFlocal", "femurLFlocal", "femurMFlocal"],
        "output": "femurVP.csv",
    },
    "Tibia": {
        "points": ["tibiaALTlocal", "tibiaAMTlocal", "tibiaLTlocal", "tibiaMTlocal"],
        "output": "tibiaVP.csv",
    },
}

# LOAD .MAT FILE
mat_data = loadmat(mat_file)

# LOOP SU FEMUR / TIBIA
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
    out_path = SAVE_DIR / cfg["output"]
    data_table = pd.DataFrame(data_dict)
    data_table.to_csv(out_path, index=False, sep=",")

    print(f"File creato: {out_path}")
