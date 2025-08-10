#!/usr/bin/env python3
# intrinsic.py — OpenCV 4.12 compatible
# Calibrate camera intrinsics from your existing calib.io ChArUco sheet.
# Workflow:
#   1) Put one clear photo of your sheet as "ref.jpg".
#   2) You will CLICK the outer 4 board corners on that photo: TL, TR, BR, BL.
#   3) Put 15–30 varied photos in images/*.jpg
#   4) Run: QT_QPA_PLATFORM=xcb python3 intrinsic.py

import os, glob, sys
import numpy as np
import cv2 as cv
import cv2.aruco as aruco

# ======= YOUR BOARD GEOMETRY (from your PDF name) =======
SQUARES_X = 8                 # total squares across
SQUARES_Y = 11                # total squares down
SQUARE_LENGTH = 0.015         # 15 mm -> meters
MARKER_LENGTH = 0.011         # 11 mm -> meters
# ========================================================

REF_IMG = "ref.jpg"
CALIB_GLOB = "images/*.jpg"
DICT_CANDIDATES = [
    aruco.DICT_4X4_50,
    aruco.DICT_4X4_100,
    aruco.DICT_4X4_250,
    aruco.DICT_4X4_1000 if hasattr(aruco, "DICT_4X4_1000") else aruco.DICT_4X_4_1000,
]

def log(m): print(m, flush=True)

# ---------- click 4 corners ----------
_clicked = []
def _on_mouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN and len(_clicked) < 4:
        _clicked.append((x, y))
        img = param.copy()
        for i,(cx,cy) in enumerate(_clicked):
            cv.circle(img, (cx,cy), 6, (0,255,0), -1)
            cv.putText(img, str(i+1), (cx+8, cy-8), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv.imshow("Click 4 board corners: TL, TR, BR, BL", img)

def click_corners(image_bgr):
    vis = image_bgr.copy()
    cv.imshow("Click 4 board corners: TL, TR, BR, BL", vis)
    cv.setMouseCallback("Click 4 board corners: TL, TR, BR, BL", _on_mouse, param=vis)
    log(">>> Click the 4 board corners in order: TL, TR, BR, BL. Then press any key.")
    cv.waitKey(0)
    cv.destroyAllWindows()
    if len(_clicked) != 4:
        sys.exit("Need exactly 4 clicks (TL, TR, BR, BL).")
    return np.array(_clicked, np.float32)  # (4,2)

# ---------- dictionary selection ----------
def pick_best_dictionary(gray):
    best_dic = None
    best_n = -1
    best_pack = None
    for d in DICT_CANDIDATES:
        dictionary = aruco.getPredefinedDictionary(d)
        detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)
        n = 0 if ids is None else len(ids)
        log(f"Trying dictionary {d}: found {n} markers")
        if n > best_n:
            best_dic, best_n, best_pack = d, n, (corners, ids, dictionary, detector)
    if best_n <= 0:
        sys.exit("No ArUco markers detected on ref.jpg with any 4x4 dictionary.")
    log(f"[OK] Dictionary chosen: {best_dic} (markers on ref: {best_n})")
    return best_pack  # (corners, ids, dictionary, detector)

# ---------- map ref marker quads to board plane ----------
def image_to_board_H(pts_img):
    Wm = SQUARES_X * SQUARE_LENGTH
    Hm = SQUARES_Y * SQUARE_LENGTH
    pts_board = np.array([[0,0],[Wm,0],[Wm,Hm],[0,Hm]], np.float32)
    H, _ = cv.findHomography(pts_img, pts_board, method=0)
    return H

def map_marker_quad_to_board(quad_img_px, H):
    """
    quad_img_px: (4,2) pixel coords of a marker in ref.jpg (as detected)
    Returns (4,3) object points in board coordinates (meters), Z=0,
    resized to exact MARKER_LENGTH around the mapped centroid.
    """
    q = np.hstack([quad_img_px, np.ones((4,1), np.float32)])  # 4x3
    qb = (H @ q.T).T
    qb = qb[:, :2] / qb[:, 2:3]  # 4x2 approx in meters on the board plane
    ctr = qb.mean(axis=0)

    # orientation from mapped quad
    v0 = qb[1] - qb[0]
    v1 = qb[3] - qb[0]
    ax = v0 / (np.linalg.norm(v0) + 1e-9)
    ay = v1 / (np.linalg.norm(v1) + 1e-9)

    half = MARKER_LENGTH / 2.0
    square = np.array([
        ctr + (-half)*ax + (-half)*ay,
        ctr + ( half)*ax + (-half)*ay,
        ctr + ( half)*ax + ( half)*ay,
        ctr + (-half)*ax + ( half)*ay
    ], np.float32)
    return np.hstack([square, np.zeros((4,1), np.float32)])   # (4,3), Z=0

# ---------- build exact board from your ref.jpg ----------
def build_custom_board_from_ref(ref_bgr):
    gray = cv.cvtColor(ref_bgr, cv.COLOR_BGR2GRAY)

    # Pick dictionary & detect markers on ref
    corners_ref, ids_ref, dictionary, detector = pick_best_dictionary(gray)
    dbg = ref_bgr.copy()
    aruco.drawDetectedMarkers(dbg, corners_ref, ids_ref)
    cv.imwrite("ref_detect_debug.jpg", dbg)

    # You click TL,TR,BR,BL on ref
    ref_for_click = dbg  # show markers as visual aid
    pts_img = click_corners(ref_for_click)

    # Homography image -> (X,Y) on board plane in meters
    H = image_to_board_H(pts_img)

    # Build Board geometry for all detected markers in ref
    objPoints_list = []
    ids_list = []
    for quad, mid in zip(corners_ref, ids_ref.flatten()):
        quad_img = quad.reshape(4,2).astype(np.float32)
        obj4x3 = map_marker_quad_to_board(quad_img, H)
        objPoints_list.append(obj4x3)
        ids_list.append(int(mid))
    ids_arr = np.array(ids_list, np.int32).reshape(-1,1)

    # Create Board (OpenCV 4.12: class constructor)
    board = aruco.Board(objPoints_list, dictionary, ids_arr)
    board_ids_list = ids_arr.flatten().tolist()
    log(f"[OK] Custom board built from ref: markers={len(board_ids_list)}")
    return board, board_ids_list, dictionary

# ---------- collect frames & format for OpenCV 4.12 ----------
def collect_frames_for_calibration(dictionary, board_ids_list):
    detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
    paths = sorted(glob.glob(CALIB_GLOB))
    if not paths:
        sys.exit(f"No images found in {CALIB_GLOB}")

    all_corners = []   # list of numpy arrays, each containing all corners for one image
    all_ids = []       # list of per-image arrays
    counter = []       # int markers per image
    img_size = None
    board_id_set = set(board_ids_list)

    for p in paths:
        img = cv.imread(p)
        if img is None:
            log(f"[SKIP] {p}"); continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = gray.shape[::-1]

        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            log(f"[{os.path.basename(p)}] markers=0"); continue

        # Filter to markers we know are on your printed sheet
        mask = [(int(i[0]) in board_id_set) for i in ids]
        if not any(mask):
            log(f"[{os.path.basename(p)}] markers found, but none match your board IDs"); continue

        corners_f = [c for c, keep in zip(corners, mask) if keep]
        ids_f = np.array([i for i, keep in zip(ids, mask) if keep], np.int32).reshape(-1,1)

        # Stack all corners for this image into a single array (n_markers, 1, 4, 2)
        if corners_f:
            # Each corner from detectMarkers is (1, 4, 2), we need to stack them properly
            corners_array = np.array(corners_f, dtype=np.float32)  # This gives us (n_markers, 1, 4, 2)
            # Ensure the shape is exactly what OpenCV expects
            if corners_array.ndim == 3:  # If we got (n_markers, 4, 2)
                corners_array = corners_array[:, np.newaxis, :, :]  # Add the missing dimension
            corners_array = corners_array.astype(np.float32)
            
            all_corners.append(corners_array)
            all_ids.append(ids_f)
            counter.append(len(ids_f))
            log(f"[{os.path.basename(p)}] kept markers: {len(ids_f)}, corners shape: {corners_array.shape}")

    if not all_corners:
        sys.exit("No usable frames after filtering.")
    return all_corners, all_ids, np.array(counter, np.int32), img_size

# ---------- calibration using standard calibrateCamera ----------
def calibrate(board, dictionary, all_corners, all_ids, counter, img_size):
    # Instead of using calibrateCameraAruco, let's use standard calibrateCamera
    # We'll convert our ArUco detections to object/image point pairs
    
    log(f"[DEBUG] Number of images: {len(all_corners)}")
    log(f"[DEBUG] Counter array: {counter}")
    
    # Prepare object points and image points for standard calibration
    object_points = []  # 3D points in real world space
    image_points = []   # 2D points in image plane
    
    # Get the object points from our board for each detected marker
    board_obj_points = {}  # marker_id -> object points
    board_ids = board.getIds().flatten()
    board_corners = board.getObjPoints()
    
    for i, marker_id in enumerate(board_ids):
        board_obj_points[marker_id] = board_corners[i]  # 4x3 array
    
    log(f"[DEBUG] Board has {len(board_obj_points)} markers")
    
    total_used_images = 0
    for img_idx, (corners_array, ids_array) in enumerate(zip(all_corners, all_ids)):
        if corners_array.shape[0] == 0:
            continue
            
        # Collect object and image points for this image
        obj_pts_this_img = []
        img_pts_this_img = []
        
        ids_flat = ids_array.flatten()
        for marker_idx, marker_id in enumerate(ids_flat):
            if marker_id in board_obj_points:
                # Get 3D object points for this marker
                obj_pts = board_obj_points[marker_id]  # (4, 3)
                obj_pts_this_img.append(obj_pts)
                
                # Get 2D image points for this marker
                img_pts = corners_array[marker_idx, 0, :, :]  # (4, 2)
                img_pts_this_img.append(img_pts)
        
        if len(obj_pts_this_img) >= 4:  # Need at least 4 markers for calibration
            object_points.append(np.vstack(obj_pts_this_img).astype(np.float32))
            image_points.append(np.vstack(img_pts_this_img).astype(np.float32))
            total_used_images += 1
            log(f"[DEBUG] Image {img_idx}: using {len(obj_pts_this_img)} markers ({len(obj_pts_this_img)*4} points)")
    
    if total_used_images < 3:
        sys.exit(f"Not enough images with sufficient markers: {total_used_images} (need at least 3)")
    
    log(f"[DEBUG] Using {total_used_images} images for calibration")
    log(f"[DEBUG] Object points shapes: {[op.shape for op in object_points[:3]]}")
    log(f"[DEBUG] Image points shapes: {[ip.shape for ip in image_points[:3]]}")

    log("[STEP] Standard camera calibration...")
    
    # Use standard calibrateCamera function
    rms, K, dist, rvecs, tvecs = cv.calibrateCamera(
        object_points,
        image_points,
        img_size,
        None,  # camera matrix
        None   # distortion coefficients
    )
    
    log(f"[RESULT] RMS reprojection error: {rms:.4f}")
    log(f"K =\n{K}")
    log(f"dist = {dist.ravel()}")

    np.save("K.npy", K)
    np.save("dist.npy", dist)
    log("[OK] Saved K.npy and dist.npy")
    
    return rms, K, dist

def main():
    log(f"[INFO] OpenCV: {cv.__version__}")

    if not os.path.exists(REF_IMG):
        sys.exit(f"Missing {REF_IMG}")
    ref = cv.imread(REF_IMG)
    if ref is None:
        sys.exit(f"Could not read {REF_IMG}")
    log(f"[INFO] {REF_IMG} loaded: {ref.shape[1]}x{ref.shape[0]}")

    # Build exact board from your printed sheet + your 4 clicks
    board, board_ids_list, dictionary = build_custom_board_from_ref(ref)

    # Collect calibration frames (marker-only, robust)
    all_corners, all_ids, counter, img_size = collect_frames_for_calibration(dictionary, board_ids_list)

    if len(counter) < 8:
        log(f"[WARN] Only {len(counter)} usable images; calibration may be poor (aim for 15–25).")

    # Run calibration
    calibrate(board, dictionary, all_corners, all_ids, counter, img_size)

if __name__ == "__main__":
    main()
