import os
import torch
import pickle
from typing import Tuple


def update_radius_in_data(
        filename: str,
        key: Tuple[int, int, int],
        new_radius: torch.Tensor,
        output: str = None):

    with open(filename, "rb") as f:
        obj = pickle.load(f)

    if not hasattr(obj, "data"):
        raise TypeError(f"Object of type {type(obj)} has no .data dict.")

    if key not in obj.data:
        raise KeyError(f"Key {key} not found in the pickle.")

    rec = obj.data[key]

    if not hasattr(rec, "radius"):
        raise TypeError(f"Record at key {key} does not have a .radius field.")

    # Confirm procedure with the user
    old_radius = rec.radius
    print(f"Current radius at key {key}: {old_radius}")
    print(f"To be overwritten with: {new_radius}")

    # Ask for confirmation
    answer = input("Do you want to overwrite this value? (y/n): ").strip().lower()

    if answer not in ("y", "n"):
        raise ValueError("Invalid answer; must be 'y' or 'n'.")

    if answer == "n":
        raise RuntimeError("Overwrite canceled by user.")

    rec._result.bound = new_radius

    out = output or filename
    with open(out, "wb") as f:
        pickle.dump(obj, f)

    print(f"Updated radius at key {key} and saved to {out}")

if __name__ == '__main__':

    file_path = r"C:\Git\ConcentrationInequalities\results\W2\gaussianmixture\dims_3\setting_0\joint_diagonal_milp"
    seed = 9
    if seed == 0:
        file_name = "data_driven_radii.pickle"
    else:
        file_name = f"data_driven_radii_seed={seed}.pickle"
    file_path = os.path.join(file_path, file_name)

    update_radius_in_data(
        filename=file_path,
        key=(5000, 1000, 1000),
        new_radius=torch.tensor(0.52)
    )
