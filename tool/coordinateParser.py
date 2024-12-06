import numpy as np

def convert_coordinates(coords, old_res, new_res):
    """
    Mengonversi koordinat dari resolusi asli ke resolusi target.

    Args:
        coords (np.ndarray): Array koordinat dengan bentuk (N, 2).
        old_res (tuple): Resolusi asli (width, height).
        new_res (tuple): Resolusi target (width, height).

    Returns:
        np.ndarray: Koordinat baru pada resolusi target.
    """
    old_width, old_height = old_res
    new_width, new_height = new_res

    scale_x = new_width / old_width
    scale_y = new_height / old_height

    # Skalakan koordinat
    converted_coords = coords * np.array([scale_x, scale_y])
    return converted_coords.astype(int)


# Koordinat asli
source_coords = np.array([
    [619, 394],
    [1032, 423],
    [968, 717],
    [240, 666]
])

# Resolusi asli dan target
old_resolution = (1280, 720)
new_resolution = (1920, 1080)

# Konversi koordinat
converted_coords = convert_coordinates(source_coords, old_resolution, new_resolution)
print("Koordinat baru:", converted_coords)
