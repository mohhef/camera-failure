import numpy as np
import matplotlib.pyplot as plt
import cv2

from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import distance

from utils import Node, StressPropagator
from visualizers import generate_glass_image,direction_comparison


np.random.seed(100)

def compute_pairwise_distances(points):
    return distance.squareform(distance.pdist(points))

def find_mst_using_prims(points):
    dist_matrix = compute_pairwise_distances(points)
    mst = minimum_spanning_tree(dist_matrix)
    return mst

def generate_glass(
        H = 375,
        W = 1242,
        BREAK_THRESHOLD = 200,
        NN_RADIUS = 15,
        IMPACT_FORCE = 1e3,
        IMPACT_ANGLE = 35,
        K = 1,
        N = 1e5,
        SUN_ANGLE=90,
        IMPACT_X=None,
        IMPACT_Y=None,
        show_plot=True
):
    N = (int)(N)
    points = np.random.uniform(low=[0,0] ,high=[H,W], size=(N,2)).astype('int')

    # Allow configurable impact point
    if IMPACT_X is not None and IMPACT_Y is not None:
        target = np.array([int(H * IMPACT_Y), int(W * IMPACT_X)])
        distances = np.linalg.norm(points - target, axis=1)
        impact_pt_idx = np.argmin(distances)
    else:
        impact_pt_idx = np.random.randint(low=0,high=N)
    IMPACT_POINT = np.array([points[impact_pt_idx]]) # Adding an extra dimension (2,) -> (1,2)
    IMPACT_ANGLE = 35

    node_features = np.zeros((N,3))
    node_features[impact_pt_idx] = np.array(
        [
            IMPACT_FORCE,
            np.cos(np.deg2rad(IMPACT_ANGLE)),
            np.sin(np.deg2rad(IMPACT_ANGLE))
        ]
    )
    broken_glass_img = np.zeros((H,W,3), dtype=np.uint8)
    all_points = {}
    for i in range(points.shape[0]):
        x,y = points[i][0],points[i][1]
        all_points[(x,y)] = Node(x,y)

    SP = StressPropagator(
            BREAK_THRESHOLD = BREAK_THRESHOLD,
            NN_RADIUS = NN_RADIUS,
            IMPACT_POINT = IMPACT_POINT,
            IMPACT_FORCE = IMPACT_FORCE,
            IMPACT_ANGLE = IMPACT_ANGLE,
            K = K,
            all_points = all_points,
            points = points,
    )
    
    pointsAll = []
    stress_vals = []
    for edge_list in SP.all_edges.values():
        for edge in edge_list:
            stress_vals.append(edge.edge_stress)
            pointsAll.append([edge.source_node.x,edge.source_node.y])
            pointsAll.append([edge.target_node.x,edge.target_node.y])
    pointsAll = np.array(pointsAll)
    stress_vals = np.array(stress_vals)

    mst = find_mst_using_prims(pointsAll)
    mst = mst.toarray().astype(float)

    broken_glass_img3 = np.zeros((H,W,3), dtype=np.uint8)
    sun_angle = SUN_ANGLE
    # Step 3: Plot the points and MST
    plt.figure(figsize=(15,4))
    for i in range(len(pointsAll)):
        for j in range(i + 1, len(pointsAll)):
            if mst[i, j] != 0:
                start_point = (pointsAll[i, 1], pointsAll[i, 0])
                end_point = (pointsAll[j, 1], pointsAll[j, 0])
                sun_angle_rad = np.radians(sun_angle)
                sun_vector = np.array([np.cos(sun_angle_rad),np.sin(sun_angle_rad)])
                angle,_ = direction_comparison(sun_direction=sun_vector,point1=start_point,point2=end_point)
                grey_value = max(0,200*(1 - abs(np.cos(np.radians(angle)))))
                line_color = (grey_value,grey_value,grey_value)
                cv2.line(broken_glass_img3, start_point, end_point, line_color, thickness=1)

    if show_plot:
        plt.imshow(broken_glass_img3)
        plt.axis('off')
        plt.show()


    if len(SP.all_edges.keys())>=2:
        img = generate_glass_image(all_edges = SP.all_edges,H=H,W=W,pts_img=broken_glass_img,sun_angle=sun_angle)
        result = np.maximum(img,broken_glass_img3)

        if show_plot:
            plt.imshow(result)
            plt.axis('off')
            plt.show()

        return result

    return broken_glass_img3

if __name__=='__main__':
    for i in range(1):
        generate_glass(
            IMPACT_FORCE = 500,
            BREAK_THRESHOLD = 300,
            NN_RADIUS = 65,
            N = 1e4,
            K = 1,
        )
