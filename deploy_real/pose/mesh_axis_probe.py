import argparse
import numpy as np


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("zero vector")
    return v / n


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe OBJ axis / long-axis direction via AABB/OBB/PCA")
    parser.add_argument("mesh", type=str, help="Path to mesh file (obj/ply/stl/...) relative or absolute")
    args = parser.parse_args()

    import trimesh

    mesh = trimesh.load(args.mesh, force="mesh")
    if not hasattr(mesh, "vertices"):
        raise RuntimeError(f"Failed to load mesh: {args.mesh}")

    verts = np.asarray(mesh.vertices)
    print(f"verts shape: {verts.shape}")

    # AABB in mesh frame
    min_xyz = verts.min(axis=0)
    max_xyz = verts.max(axis=0)
    ext_aabb = max_xyz - min_xyz
    aabb_long = int(np.argmax(ext_aabb))

    print("=== AABB (in OBJ frame) ===")
    print("min xyz:", min_xyz)
    print("max xyz:", max_xyz)
    print("extents xyz:", ext_aabb)
    print("longest axis idx (0=x,1=y,2=z):", aabb_long)

    # Oriented bounding box: extents are in OBB frame; to_origin maps mesh->OBB frame
    to_origin, ext_obb = trimesh.bounds.oriented_bounds(mesh)
    obb_long = int(np.argmax(ext_obb))

    # OBB axes in OBJ frame can be read from inverse(to_origin)
    # to_origin maps points from OBJ frame into OBB frame.
    # So R_obj_to_obb = to_origin[:3,:3]; hence OBB axes in OBJ frame are columns of R_obj_to_obb.T
    R_obj_to_obb = np.asarray(to_origin[:3, :3])
    R_obb_to_obj = R_obj_to_obb.T
    obb_axes_in_obj = R_obb_to_obj  # columns are OBB x/y/z axes expressed in OBJ frame

    print("\n=== OBB (oriented bounds) ===")
    print("extents (obb x,y,z):", ext_obb)
    print("longest obb axis idx (0=x,1=y,2=z):", obb_long)
    print("OBB axes expressed in OBJ frame (columns are x,y,z):\n", obb_axes_in_obj)

    # PCA on vertices (principal directions)
    V = verts - verts.mean(axis=0, keepdims=True)
    C = (V.T @ V) / max(len(V) - 1, 1)
    vals, vecs = np.linalg.eigh(C)  # ascending
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    v0 = _normalize(vecs[:, 0])

    print("\n=== PCA (on vertices) ===")
    print("eigenvalues (desc):", vals)
    print("principal axis v0 (OBJ frame):", v0)
    print("note: sign is arbitrary; v0 and -v0 are same axis")

    # A very pragmatic recommendation:
    # - If AABB is clearly dominant (ratio>1.2), use that axis.
    # - Else use PCA v0.
    ext_sorted = np.sort(ext_aabb)[::-1]
    ratio = float(ext_sorted[0] / max(ext_sorted[1], 1e-12))

    print("\n=== Recommendation ===")
    print("AABB dominance ratio (longest/2nd):", ratio)
    if ratio > 1.2:
        axis_names = ["+X", "+Y", "+Z"]
        print("Recommended long axis: OBJ frame", axis_names[aabb_long], "(AABB-dominant)")
    else:
        print("Recommended long axis: PCA v0 (OBJ frame)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
