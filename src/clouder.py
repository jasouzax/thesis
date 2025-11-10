import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_ply(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse header to find vertex count and end of header
    vertex_count = 0
    header_end = 0
    has_colors = False
    for i, line in enumerate(lines):
        if line.startswith('element vertex'):
            vertex_count = int(line.split()[2])
        elif line.startswith('property uchar red'):
            has_colors = True
        elif line.startswith('end_header'):
            header_end = i + 1
            break
    
    # Read points (and colors if present)
    points = []
    colors = []
    for line in lines[header_end:header_end + vertex_count]:
        parts = line.strip().split()
        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        points.append([x, y, z])
        if has_colors:
            r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
            colors.append([r / 255.0, g / 255.0, b / 255.0])  # Normalize to [0,1]
    
    points = np.array(points)
    colors = np.array(colors) if colors else None
    return points, colors

# Load the point cloud
points, colors = read_ply('point_cloud.ply')

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors if colors is not None else 'b', s=1)  # s=1 for small points

# Set labels and aspect ratio
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_box_aspect((np.ptp(points[:,0]), np.ptp(points[:,1]), np.ptp(points[:,2])))  # Equal aspect ratio

plt.show()