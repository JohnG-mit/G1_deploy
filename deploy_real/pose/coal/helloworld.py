import os
import sys
parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent2_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import coal
# Optional:
# The Pinocchio library is a rigid body algorithms library and has a handy SE3 module.
# It can be installed as simply as `conda -c conda-forge install pinocchio`.
# Installing pinocchio also installs coal.
import pinocchio as pin


def loadConvexMesh(file_name: str):
    loader = coal.MeshLoader()
    bvh: coal.BVHModelBase = loader.load(file_name)
    bvh.buildConvexHull(True, "Qt")
    return bvh.convex

if __name__ == "__main__":
    # Create coal shapes
    shape1 = coal.Ellipsoid(0.7, 1.0, 0.8)
    shape2 = loadConvexMesh("/home/johng/repo/G1_deploy/deploy_real/pose/mesh/stool-leg-scaled.obj")

    # Define the shapes' placement in 3D space
    T1 = coal.Transform3s()
    T1.setTranslation(pin.SE3.Random().translation)
    T1.setRotation(pin.SE3.Random().rotation)
    T2 = coal.Transform3s()
    # Using np arrays also works
    T1.setTranslation(np.random.rand(3))
    T2.setRotation(pin.SE3.Random().rotation)

    # Define collision requests and results
    col_req = coal.CollisionRequest()
    col_res = coal.CollisionResult()

    # Collision call
    coal.collide(shape1, T1, shape2, T2, col_req, col_res)

    # Accessing the collision result once it has been populated
    print("Is collision? ", {col_res.isCollision()})
    if col_res.isCollision():
        contact: coal.Contact = col_res.getContact(0)
        print("Penetration depth: ", contact.penetration_depth)
        print("Distance between the shapes including the security margin: ", contact.penetration_depth + col_req.security_margin)
        print("Witness point shape1: ", contact.getNearestPoint1())
        print("Witness point shape2: ", contact.getNearestPoint2())
        print("Normal: ", contact.normal)

    # Before running another collision call, it is important to clear the old one
    col_res.clear()