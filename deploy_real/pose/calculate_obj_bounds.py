import numpy as np

def calculate_bounds(obj_path):
    vertices = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    vertices = np.array(vertices)
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    center = (min_coords + max_coords) / 2
    dimensions = max_coords - min_coords
    
    print(f"Min: {min_coords}")
    print(f"Max: {max_coords}")
    print(f"Center: {center}")
    print(f"Dimensions: {dimensions}")
    return center

if __name__ == "__main__":
    calculate_bounds("/home/johng/repo/G1_deploy/deploy_real/pose/mesh/stool-leg-scaled.obj")
