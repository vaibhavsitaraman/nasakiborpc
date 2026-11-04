import cv2
import numpy as np
import os
import glob
from pathlib import Path


class CameraCorrectionModel:
    def __init__(self, input_folder="input", output_folder="output"):
        self.input_folder = input_folder
        self.output_folder = output_folder

        # Create output folder if it doesn't exist
        Path(self.output_folder).mkdir(exist_ok=True)

        # Camera intrinsics and distortion coefficients
        self.camera_matrix = np.array(
            [[523.105750, 0.000000, 635.434258], [0.000000, 534.765913, 500.335102], [0.000000, 0.000000, 1.000000]],
            dtype=np.float32)

        self.dist_coeffs = np.array([-0.164787, 0.020375, -0.001572, -0.000369, 0.000000], dtype=np.float32)

        # ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # Sheet dimensions (in cm) - used for ratios
        self.sheet_width = 27.0
        self.sheet_height = 15.0
        self.aruco_size = 5.0
        self.aruco_offset_x = 1.75  # from right edge
        self.aruco_offset_y = 1.75  # from top edge

        # ArUco marker size in output image (pixels)
        self.target_aruco_size = 400

    def preprocess_image(self, image, filename):
        """Step 1: Preprocessing - undistort image"""
        print(f"Step 1: Preprocessing {filename}")

        # Undistort image
        undistorted = cv2.undistort(image, self.camera_matrix, self.dist_coeffs)

        # Save checkpoint
        cv2.imwrite(os.path.join(self.output_folder, f"{filename}_1_undistorted.jpg"), undistorted)

        return undistorted

    def detect_and_orient_aruco(self, image, filename):
        """Detect ArUco marker and determine correct orientation"""
        print(f"Detecting ArUco marker in {filename}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        corners, ids, _ = detector.detectMarkers(gray)

        if len(corners) == 0:
            raise ValueError(f"No ArUco marker detected in {filename}")

        # Use the first detected marker
        aruco_corners = corners[0][0]

        # Draw detected marker for visualization
        marked_image = image.copy()
        cv2.aruco.drawDetectedMarkers(marked_image, corners, ids)
        cv2.imwrite(os.path.join(self.output_folder, f"{filename}_2a_aruco_detected.jpg"), marked_image)

        return aruco_corners, image

    def rotate_image_to_correct_orientation(self, image, aruco_corners, filename):
        """Rotate image so ArUco marker is in correct orientation (top-right)"""
        print(f"Orienting image {filename}")

        # Calculate the orientation of the ArUco marker
        # ArUco corners are ordered: top-left, top-right, bottom-right, bottom-left

        # Get the top edge vector (from top-left to top-right)
        top_left = aruco_corners[0]
        top_right = aruco_corners[1]
        top_vector = top_right - top_left

        # Calculate angle of the top edge relative to horizontal
        angle_rad = np.arctan2(top_vector[1], top_vector[0])
        angle_deg = np.degrees(angle_rad)

        # Normalize angle to [-180, 180] range
        while angle_deg > 180:
            angle_deg -= 360
        while angle_deg < -180:
            angle_deg += 360

        # Determine the rotation needed to make the marker upright
        # We want the top edge to be horizontal (0 degrees)
        rotation_angle = angle_deg

        print(f"ArUco top edge angle: {angle_deg:.1f}°, applying rotation: {rotation_angle:.1f}°")

        # Get image dimensions
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # Calculate rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

        # Calculate new image size to ensure no content is lost
        cos_abs = np.abs(rotation_matrix[0, 0])
        sin_abs = np.abs(rotation_matrix[0, 1])
        new_w = int(h * sin_abs + w * cos_abs)
        new_h = int(h * cos_abs + w * sin_abs)

        # Ensure new size is at least as large as original
        new_w = max(new_w, w)
        new_h = max(new_h, h)

        # Adjust rotation matrix for new center
        rotation_matrix[0, 2] += (new_w - w) / 2
        rotation_matrix[1, 2] += (new_h - h) / 2

        # Rotate image with expanded canvas to preserve all content
        rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h),
                                 borderMode=cv2.BORDER_TRANSPARENT)

        # Rotate ArUco corners accordingly
        ones = np.ones((4, 1))
        aruco_corners_homo = np.hstack([aruco_corners, ones])
        rotated_aruco_corners = rotation_matrix.dot(aruco_corners_homo.T).T

        # Verify the rotation worked by checking if the marker is roughly upright
        new_top_vector = rotated_aruco_corners[1] - rotated_aruco_corners[0]
        new_angle = np.degrees(np.arctan2(new_top_vector[1], new_top_vector[0]))
        print(f"After rotation, ArUco top edge angle: {new_angle:.1f}°")

        cv2.imwrite(os.path.join(self.output_folder, f"{filename}_2b_rotated.jpg"), rotated)

        return rotated, rotated_aruco_corners

    def debug_edge_detection(self, image, filename, step_name):
        """Create debug visualizations for edge detection process"""
        debug_folder = os.path.join(self.output_folder, "debug_edges")
        Path(debug_folder).mkdir(exist_ok=True)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(debug_folder, f"{filename}_{step_name}_01_gray.jpg"), gray)

        # Try different preprocessing techniques

        # 1. Simple blur + Canny
        blurred1 = cv2.GaussianBlur(gray, (5, 5), 0)
        edges1 = cv2.Canny(blurred1, 30, 100, apertureSize=3)
        cv2.imwrite(os.path.join(debug_folder, f"{filename}_{step_name}_02_canny_basic.jpg"), edges1)

        # 2. Adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imwrite(os.path.join(debug_folder, f"{filename}_{step_name}_03_adaptive_thresh.jpg"), adaptive)

        # 3. Strong edge enhancement before Canny
        enhanced = cv2.addWeighted(gray, 1.5, cv2.GaussianBlur(gray, (0, 0), 1.0), -0.5, 0)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        edges2 = cv2.Canny(enhanced, 20, 60)
        cv2.imwrite(os.path.join(debug_folder, f"{filename}_{step_name}_04_canny_enhanced.jpg"), edges2)

        # 4. Multiple Canny thresholds
        edges_low = cv2.Canny(blurred1, 15, 45)
        edges_high = cv2.Canny(blurred1, 50, 150)
        cv2.imwrite(os.path.join(debug_folder, f"{filename}_{step_name}_05_canny_low_thresh.jpg"), edges_low)
        cv2.imwrite(os.path.join(debug_folder, f"{filename}_{step_name}_06_canny_high_thresh.jpg"), edges_high)

        # 5. Laplacian edge detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        cv2.imwrite(os.path.join(debug_folder, f"{filename}_{step_name}_07_laplacian.jpg"), laplacian)

        # 6. Sobel edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = np.sqrt(sobelx ** 2 + sobely ** 2)
        sobel_combined = np.uint8(sobel_combined * 255 / np.max(sobel_combined))
        cv2.imwrite(os.path.join(debug_folder, f"{filename}_{step_name}_08_sobel.jpg"), sobel_combined)

        # Return the best edge image for further processing
        return edges2  # Enhanced Canny seems most promising

    def debug_contour_detection(self, edges, original_image, filename, step_name):
        """Debug contour detection process"""
        debug_folder = os.path.join(self.output_folder, "debug_contours")
        Path(debug_folder).mkdir(exist_ok=True)

        # Try different morphological operations
        kernel_small = np.ones((2, 2), np.uint8)
        kernel_medium = np.ones((3, 3), np.uint8)
        kernel_large = np.ones((5, 5), np.uint8)

        # Closing to connect nearby edges
        closed1 = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_small)
        closed2 = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_medium)
        closed3 = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_large)

        cv2.imwrite(os.path.join(debug_folder, f"{filename}_{step_name}_01_closed_small.jpg"), closed1)
        cv2.imwrite(os.path.join(debug_folder, f"{filename}_{step_name}_02_closed_medium.jpg"), closed2)
        cv2.imwrite(os.path.join(debug_folder, f"{filename}_{step_name}_03_closed_large.jpg"), closed3)

        # Dilation to strengthen edges
        dilated1 = cv2.dilate(edges, kernel_small, iterations=1)
        dilated2 = cv2.dilate(edges, kernel_medium, iterations=2)

        cv2.imwrite(os.path.join(debug_folder, f"{filename}_{step_name}_04_dilated_light.jpg"), dilated1)
        cv2.imwrite(os.path.join(debug_folder, f"{filename}_{step_name}_05_dilated_strong.jpg"), dilated2)

        # Try different edge images for contour detection
        edge_variants = [
            ("original", edges),
            ("closed_medium", closed2),
            ("dilated_light", dilated1),
            ("dilated_strong", dilated2)
        ]

        all_contours = {}

        for variant_name, edge_img in edge_variants:
            # Find contours
            contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_contours[variant_name] = contours

            # Visualize all contours
            contour_vis = original_image.copy()
            cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(debug_folder, f"{filename}_{step_name}_contours_{variant_name}.jpg"), contour_vis)

            print(f"Found {len(contours)} contours using {variant_name} edges")

        return all_contours

    def analyze_and_score_contours(self, contours_dict, original_image, filename, step_name,
                                   expected_sheet_corners=None):
        """Analyze contours and score them for sheet detection"""
        debug_folder = os.path.join(self.output_folder, "debug_scoring")
        Path(debug_folder).mkdir(exist_ok=True)

        crop_area = original_image.shape[0] * original_image.shape[1]
        min_area = crop_area * 0.2  # At least 20% of crop area
        max_area = crop_area * 0.9  # At most 90% of crop area

        expected_ratio = self.sheet_width / self.sheet_height

        best_contours = []

        for variant_name, contours in contours_dict.items():
            print(f"\nAnalyzing {variant_name} contours:")

            scored_contours = []

            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)

                if min_area < area < max_area:
                    # Calculate various metrics
                    perimeter = cv2.arcLength(contour, True)

                    # Try different epsilon values for polygon approximation
                    for epsilon_factor in [0.01, 0.02, 0.03, 0.05]:
                        epsilon = epsilon_factor * perimeter
                        approx = cv2.approxPolyDP(contour, epsilon, True)

                        if len(approx) >= 4:
                            # Calculate rectangularity
                            rect_area = cv2.contourArea(approx)
                            bbox = cv2.boundingRect(approx)
                            bbox_area = bbox[2] * bbox[3]
                            rectangularity = rect_area / bbox_area if bbox_area > 0 else 0

                            # Calculate aspect ratio score
                            actual_ratio = bbox[2] / bbox[3] if bbox[3] > 0 else 0
                            ratio_score = 1.0 - min(abs(expected_ratio - actual_ratio) / expected_ratio, 1.0)

                            # Calculate convexity
                            hull = cv2.convexHull(approx)
                            hull_area = cv2.contourArea(hull)
                            convexity = rect_area / hull_area if hull_area > 0 else 0

                            # Combined score
                            area_score = min(area / max_area, 1.0)
                            score = area_score * rectangularity * ratio_score * convexity

                            scored_contours.append({
                                'contour': approx,
                                'score': score,
                                'area': area,
                                'rectangularity': rectangularity,
                                'ratio_score': ratio_score,
                                'convexity': convexity,
                                'epsilon_factor': epsilon_factor,
                                'bbox': bbox,
                                'variant': variant_name
                            })

            # Sort by score and keep top candidates
            scored_contours.sort(key=lambda x: x['score'], reverse=True)
            best_contours.extend(scored_contours[:3])  # Top 3 from each variant

            # Visualize top contours for this variant
            if scored_contours:
                vis_image = original_image.copy()
                for j, item in enumerate(scored_contours[:5]):  # Show top 5
                    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)][j]
                    cv2.drawContours(vis_image, [item['contour']], -1, color, 3)

                    # Add score text
                    x, y, w, h = item['bbox']
                    cv2.putText(vis_image, f"#{j + 1} S:{item['score']:.3f}",
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    print(f"  Contour {j + 1}: Score={item['score']:.3f}, "
                          f"Area={item['area']:.0f}, Rect={item['rectangularity']:.3f}, "
                          f"Ratio={item['ratio_score']:.3f}, Conv={item['convexity']:.3f}, "
                          f"Eps={item['epsilon_factor']}")

                cv2.imwrite(os.path.join(debug_folder, f"{filename}_{step_name}_scored_{variant_name}.jpg"), vis_image)

        # Sort all candidates by score
        best_contours.sort(key=lambda x: x['score'], reverse=True)

        # Create final comparison image
        if best_contours:
            final_vis = original_image.copy()
            for j, item in enumerate(best_contours[:10]):  # Show top 10 overall
                color_intensity = max(50, 255 - j * 20)
                color = (0, color_intensity, 0) if j == 0 else (color_intensity, 0, 0)
                cv2.drawContours(final_vis, [item['contour']], -1, color, 2)

                x, y, w, h = item['bbox']
                cv2.putText(final_vis, f"#{j + 1}({item['variant'][:3]}): {item['score']:.3f}",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            cv2.imwrite(os.path.join(debug_folder, f"{filename}_{step_name}_final_candidates.jpg"), final_vis)

        return best_contours[0] if best_contours else None

    def detect_sheet_corners(self, image, aruco_corners, filename):
        """Enhanced sheet corner detection with comprehensive debugging"""
        print(f"Attempting to detect sheet corners in {filename}")

        # Calculate the scale and expected sheet position based on ArUco marker
        aruco_center = np.mean(aruco_corners, axis=0)
        aruco_width = np.linalg.norm(aruco_corners[1] - aruco_corners[0])
        aruco_height = np.linalg.norm(aruco_corners[3] - aruco_corners[0])
        avg_aruco_size_px = (aruco_width + aruco_height) / 2
        scale_factor = self.aruco_size / avg_aruco_size_px

        # Calculate sheet dimensions in pixels
        sheet_width_px = self.sheet_width / scale_factor
        sheet_height_px = self.sheet_height / scale_factor

        # Calculate sheet position based on ArUco marker position
        aruco_to_right_edge_px = self.aruco_offset_x / scale_factor
        aruco_to_top_edge_px = self.aruco_offset_y / scale_factor

        # Estimate sheet corners based on ArUco position
        sheet_top_right_x = aruco_center[0] + aruco_to_right_edge_px
        sheet_top_right_y = aruco_center[1] - aruco_to_top_edge_px

        estimated_corners = np.array([
            [sheet_top_right_x - sheet_width_px, sheet_top_right_y],  # top-left
            [sheet_top_right_x, sheet_top_right_y],  # top-right
            [sheet_top_right_x, sheet_top_right_y + sheet_height_px],  # bottom-right
            [sheet_top_right_x - sheet_width_px, sheet_top_right_y + sheet_height_px]  # bottom-left
        ], dtype=np.float32)

        print(f"Estimated sheet size: {sheet_width_px:.1f}x{sheet_height_px:.1f} pixels")
        print(f"Scale factor: {scale_factor:.4f} cm/pixel")

        # Create a generous crop region around the estimated sheet
        padding_factor = 0.5  # 50% padding
        padding_x = int(sheet_width_px * padding_factor)
        padding_y = int(sheet_height_px * padding_factor)

        x_min = max(0, int(min(estimated_corners[:, 0]) - padding_x))
        y_min = max(0, int(min(estimated_corners[:, 1]) - padding_y))
        x_max = min(image.shape[1], int(max(estimated_corners[:, 0]) + padding_x))
        y_max = min(image.shape[0], int(max(estimated_corners[:, 1]) + padding_y))

        if x_max <= x_min or y_max <= y_min:
            print(f"Invalid crop region for {filename}, cannot detect sheet corners")
            return None

        print(f"Crop region: {x_min},{y_min} to {x_max},{y_max} (size: {x_max - x_min}x{y_max - y_min})")

        cropped = image[y_min:y_max, x_min:x_max]

        # Save crop region with estimated corners
        crop_vis = cropped.copy()
        adjusted_corners = estimated_corners - np.array([x_min, y_min])
        for i, corner in enumerate(adjusted_corners.astype(int)):
            if 0 <= corner[0] < crop_vis.shape[1] and 0 <= corner[1] < crop_vis.shape[0]:
                cv2.circle(crop_vis, tuple(corner), 8, (255, 0, 255), -1)
                cv2.putText(crop_vis, f'E{i}', tuple(corner + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        cv2.imwrite(os.path.join(self.output_folder, f"{filename}_2c_crop_with_estimates.jpg"), crop_vis)

        # Debug edge detection
        best_edges = self.debug_edge_detection(cropped, filename, "crop")

        # Debug contour detection
        contours_dict = self.debug_contour_detection(best_edges, cropped, filename, "crop")

        # Analyze and score contours
        best_contour = self.analyze_and_score_contours(contours_dict, cropped, filename, "crop", adjusted_corners)

        if best_contour is not None:
            # Get the corner points
            corners = best_contour['contour'].reshape(-1, 2).astype(np.float32)

            # If more than 4 points, find the 4 most extreme ones
            if len(corners) > 4:
                corners = self.find_extreme_points(corners)

            # Sort corners properly
            corners = self.sort_corners(corners)

            # Adjust for crop offset
            corners += np.array([x_min, y_min])

            # Save final visualization
            final_vis = image.copy()

            # Draw estimated corners
            for i, corner in enumerate(estimated_corners.astype(int)):
                cv2.circle(final_vis, tuple(corner), 8, (255, 0, 255), -1)
                cv2.putText(final_vis, f'E{i}', tuple(corner + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # Draw detected corners
            for i, corner in enumerate(corners.astype(int)):
                cv2.circle(final_vis, tuple(corner), 10, (0, 255, 0), -1)
                cv2.putText(final_vis, f'D{i}', tuple(corner + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw crop region
            cv2.rectangle(final_vis, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

            cv2.imwrite(os.path.join(self.output_folder, f"{filename}_2c_sheet_corners_final.jpg"), final_vis)

            print(f"Successfully detected sheet corners for {filename} with score {best_contour['score']:.3f}")
            return corners

        print(f"Could not reliably detect sheet corners for {filename}")
        return None

    def find_extreme_points(self, points):
        """Find the 4 most extreme points from a set of points"""
        if len(points) <= 4:
            return points

        # Find extreme points in 4 directions
        top_left = points[np.argmin(points[:, 0] + points[:, 1])]
        top_right = points[np.argmax(points[:, 0] - points[:, 1])]
        bottom_right = points[np.argmax(points[:, 0] + points[:, 1])]
        bottom_left = points[np.argmax(-points[:, 0] + points[:, 1])]

        return np.array([top_left, top_right, bottom_right, bottom_left])

    def sort_corners(self, corners):
        """Sort corners in order: top-left, top-right, bottom-right, bottom-left"""
        # Calculate centroid
        centroid = np.mean(corners, axis=0)

        # Sort by angle from centroid
        angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
        sorted_indices = np.argsort(angles)
        sorted_corners = corners[sorted_indices]

        # Find top-left corner (smallest sum of coordinates)
        sums = np.sum(sorted_corners, axis=1)
        top_left_idx = np.argmin(sums)

        # Reorder starting from top-left
        reordered = np.roll(sorted_corners, -top_left_idx, axis=0)

        return reordered

    def apply_homography(self, image, aruco_corners, filename):
        """Step 3: Apply homography transformation using ArUco corners and sheet corners if available"""
        print(f"Step 3: Applying homography to {filename}")

        # Calculate output image dimensions based on sheet proportions
        pixels_per_cm = self.target_aruco_size / self.aruco_size  # pixels per cm in output

        target_width = int(self.sheet_width * pixels_per_cm)
        target_height = int(self.sheet_height * pixels_per_cm)

        print(f"Target image size: {target_width}x{target_height} pixels")

        # Calculate target ArUco marker position based on known offsets
        aruco_offset_x_px = int(self.aruco_offset_x * pixels_per_cm)
        aruco_offset_y_px = int(self.aruco_offset_y * pixels_per_cm)

        # ArUco marker position in target image (from top-right corner of sheet)
        aruco_right = target_width - aruco_offset_x_px
        aruco_left = aruco_right - self.target_aruco_size
        aruco_top = aruco_offset_y_px
        aruco_bottom = aruco_top + self.target_aruco_size

        # Target ArUco corners
        target_aruco_corners = np.array([
            [aruco_left, aruco_top],  # top-left of ArUco
            [aruco_right, aruco_top],  # top-right of ArUco
            [aruco_right, aruco_bottom],  # bottom-right of ArUco
            [aruco_left, aruco_bottom]  # bottom-left of ArUco
        ], dtype=np.float32)

        # Target sheet corners
        target_sheet_corners = np.array([
            [0, 0],  # top-left
            [target_width, 0],  # top-right
            [target_width, target_height],  # bottom-right
            [0, target_height]  # bottom-left
        ], dtype=np.float32)

        # Try to detect sheet corners
        sheet_corners = self.detect_sheet_corners(image, aruco_corners, filename)

        if sheet_corners is not None:
            print(f"Using both ArUco and sheet corners for homography")
            # Use both ArUco and sheet corners for better accuracy
            source_points = np.vstack([aruco_corners.astype(np.float32), sheet_corners])
            target_points = np.vstack([target_aruco_corners, target_sheet_corners])

            # Use RANSAC to handle potential outliers
            homography_matrix, mask = cv2.findHomography(
                source_points,
                target_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0
            )

            if homography_matrix is None:
                print(f"RANSAC failed, falling back to ArUco-only homography")
                homography_matrix = cv2.getPerspectiveTransform(aruco_corners.astype(np.float32), target_aruco_corners)
        else:
            print(f"Using ArUco-only homography")
            # Fallback to ArUco corners only
            homography_matrix = cv2.getPerspectiveTransform(aruco_corners.astype(np.float32), target_aruco_corners)

        # Apply perspective transformation
        warped = cv2.warpPerspective(image, homography_matrix, (target_width, target_height))

        # Draw corners on original image for visualization
        corner_vis = image.copy()

        # Draw ArUco corners
        for i, corner in enumerate(aruco_corners.astype(int)):
            cv2.circle(corner_vis, tuple(corner), 8, (255, 0, 0), -1)
            cv2.putText(corner_vis, f'A{i}', tuple(corner + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw sheet corners if detected
        if sheet_corners is not None:
            for i, corner in enumerate(sheet_corners.astype(int)):
                cv2.circle(corner_vis, tuple(corner), 8, (0, 255, 0), -1)
                cv2.putText(corner_vis, f'S{i}', tuple(corner + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imwrite(os.path.join(self.output_folder, f"{filename}_3a_all_corners.jpg"), corner_vis)
        cv2.imwrite(os.path.join(self.output_folder, f"{filename}_3b_warped.jpg"), warped)

        return warped

    def enhance_image(self, image, filename):
        """Step 4: Greyscale enhancement with edge sharpening"""
        print(f"Step 4: Enhancing image {filename}")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(self.output_folder, f"{filename}_4a_grayscale.jpg"), gray)

        # Sharpen the image to enhance edges and details - More extreme sharpening
        # Method 1: High-pass filter for extreme edge enhancement
        gaussian_blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
        high_pass = cv2.addWeighted(gray, 2.0, gaussian_blur, -1.0, 0)

        # Method 2: Laplacian sharpening for even more edge enhancement
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpened = np.clip(gray.astype(np.float64) + 0.8 * laplacian, 0, 255).astype(np.uint8)

        # Method 3: Multiple unsharp mask iterations for extreme sharpening
        temp = gray.copy()
        for i in range(2):  # Apply unsharp mask twice
            blurred = cv2.GaussianBlur(temp, (0, 0), 1.5)
            temp = cv2.addWeighted(temp, 2.0, blurred, -1.0, 0)
            temp = np.clip(temp, 0, 255).astype(np.uint8)

        # Use the multiple unsharp mask result (most extreme but controlled)
        sharpened = temp
        cv2.imwrite(os.path.join(self.output_folder, f"{filename}_4b_sharpened.jpg"), sharpened)

        # Otsu binarization on the sharpened image
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(os.path.join(self.output_folder, f"{filename}_4c_binary.jpg"), binary)

        # Opening morphology (small kernel to preserve details)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        cv2.imwrite(os.path.join(self.output_folder, f"{filename}_4d_opened.jpg"), opened)

        # Crop out the rightmost 7cm of the sheet (leaving 20cm width)
        h, w = opened.shape
        pixels_per_cm = self.target_aruco_size / self.aruco_size  # 400px / 5cm = 80 pixels per cm

        # Calculate the final width we want (27cm - 7cm = 20cm)
        final_width_cm = self.sheet_width - 7.0  # 27 - 7 = 20cm
        final_width_px = int(final_width_cm * pixels_per_cm)  # 20cm * 80px/cm = 1600px

        # Crop the image to keep only the leftmost 20cm worth of pixels
        final = opened[:, :final_width_px]

        cv2.imwrite(os.path.join(self.output_folder, f"{filename}_4e_final.jpg"), final)

        return final

    def process_single_image(self, image_path):
        """Process a single image through the entire pipeline"""
        filename = Path(image_path).stem
        print(f"\n=== Processing {filename} ===")

        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image {image_path}")
                return None

            # Step 1: Preprocessing
            undistorted = self.preprocess_image(image, filename)

            # Step 2: Detect and orient ArUco marker
            aruco_corners, oriented_image = self.detect_and_orient_aruco(undistorted, filename)
            rotated_image, rotated_aruco_corners = self.rotate_image_to_correct_orientation(
                oriented_image, aruco_corners, filename)

            # Step 3: Apply homography (with sheet corner detection attempt)
            warped = self.apply_homography(rotated_image, rotated_aruco_corners, filename)

            # Step 4: Enhancement
            final = self.enhance_image(warped, filename)

            print(f"Successfully processed {filename}")
            return final

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def process_all_images(self):
        """Process all images in the input folder"""
        # Supported image extensions
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']

        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(self.input_folder, ext)))
            image_files.extend(glob.glob(os.path.join(self.input_folder, ext.upper())))

        if not image_files:
            print(f"No image files found in {self.input_folder}")
            return

        print(f"Found {len(image_files)} image(s) to process")

        for image_path in image_files:
            self.process_single_image(image_path)

        print(f"\nProcessing complete. Check {self.output_folder} for results.")
        print(f"Debug images can be found in:")
        print(f"  - {self.output_folder}/debug_edges/")
        print(f"  - {self.output_folder}/debug_contours/")
        print(f"  - {self.output_folder}/debug_scoring/")


def main():
    """Main function to run the camera correction model"""
    # Initialize the model
    model = CameraCorrectionModel(input_folder="input", output_folder="output")

    # Process all images
    model.process_all_images()


if __name__ == "__main__":
    main()