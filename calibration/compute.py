import cv2
import numpy as np

# Load your calibration
K = np.load("K.npy")
dist = np.load("dist.npy")

# These are known printer coordinates (in mm)
# and their corresponding pixel coordinates in the image
# You must measure these manually once
printer_points = np.array([
    [42, 192],       # front-left of bed
    [190, 188],     # front-right
    [188, 99],   # back-right
    [40, 104]      # back-left
], dtype=np.float32)

pixel_points = np.array([
    [155, 217],   # pixel position for (0, 0)
    [1282, 309],  # pixel position for (200, 0)
    [1231, 994],  # pixel position for (200, 200)
    [63, 892]    # pixel position for (0, 200)
], dtype=np.float32)

# Undistort pixel points before computing homography
pixel_points = cv2.undistortPoints(
    pixel_points.reshape(-1, 1, 2), K, dist, P=K
).reshape(-1, 2)

# Compute homography (printer -> pixel)
H, _ = cv2.findHomography(printer_points, pixel_points)

# Inverse homography (pixel -> printer)
H_inv, _ = cv2.findHomography(pixel_points, printer_points)

def printer_to_px(x, y):
    pt = np.array([x, y, 1.0])
    uv = H @ pt
    return (uv[0] / uv[2], uv[1] / uv[2])

def px_to_printer(u, v):
    pt = np.array([u, v, 1.0])
    xy = H_inv @ pt
    return (xy[0] / xy[2], xy[1] / xy[2])


print("Inverse Homography (pixel -> printer):")
print(H_inv)
print("Homography (printer -> pixel):")
print(H)

# Example usage:
print("Printer (100, 100) -> pixel", printer_to_px(100, 100))
print("Pixel (960, 540) -> printer", px_to_printer(960, 540))
print("Pixel (960, 540) -> printer", px_to_printer(681, 130))
print("Pixel (960, 540) -> printer", px_to_printer(663, 151))



