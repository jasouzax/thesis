import cv2
import torch
import numpy as np

# Set up camera
cap = cv2.VideoCapture(0)

# Set resolution to 5MP if possible (may need adjustment based on camera capabilities)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)

# Load MiDaS model
model_type = "MiDaS_small"  # Use "DPT_Large" or "DPT_Hybrid" for better accuracy but slower performance
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# Load transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type == "MiDaS_small":
    transform = midas_transforms.small_transform
else:
    transform = midas_transforms.dpt_transform  # For DPT models

# Function to write point cloud to PLY file
def write_ply(filename, points, colors=None):
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        if colors is not None:
            for p, c in zip(points, colors):
                f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")
        else:
            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")

# Capture frame
ret, frame = cap.read()
if not ret:
    print("Failed to capture image")
    cap.release()
    exit()

height, width = frame.shape[:2]
print(f"Captured resolution: {width}x{height}")

# Preprocess image
img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
imgbatch = transform(img).to(device)

# Compute depth
with torch.no_grad():
    prediction = midas(imgbatch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

disparity = prediction.cpu().numpy()

# Convert disparity to depth (relative depth, scale is arbitrary)
depth = 1.0 / disparity  # Higher values mean farther away

# Normalize for visualization (optional)
depth_norm = cv2.normalize(depth, None, 0, 1, norm_type=cv2.NORM_MINMAX)
cv2.imshow("Depth Map", depth_norm)
cv2.waitKey(0)

# Camera intrinsics estimation (assuming pinhole model and horizontal FOV=160 degrees)
# Note: For fisheye lenses (common with 160° FOV), this is approximate. Consider undistorting the image first for accuracy.
hfov_deg = 160
hfov_rad = np.deg2rad(hfov_deg)
fx = width / (2 * np.tan(hfov_rad / 2))
fy = fx  # Assuming square pixels and horizontal FOV; adjust if vertical FOV is known
cx = width / 2
cy = height / 2

# Generate point cloud
u, v = np.meshgrid(np.arange(width), np.arange(height))
x = (u - cx) * depth / fx
y = (v - cy) * depth / fy
z = depth

# Flatten to point list
points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)

# Optional: Colors from original image
colors = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).reshape(-1, 3)

# Save to PLY file (with colors)
write_ply("point_cloud.ply", points, colors)

print("Point cloud saved to point_cloud.ply")

# Release resources
cap.release()
cv2.destroyAllWindows()