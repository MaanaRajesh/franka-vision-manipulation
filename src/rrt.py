import sys
sys.path.append("/home/student/meam520_ws/src/meam520_labs")
import numpy as np
import random
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap
from copy import deepcopy
from lib.calculateFK import FK

def isRobotCollided(q, obstacles):
    """
    Check if robot configuration collides with obstacles
    :param q: robot configuration (1x7 array)
    :param obstacles: array of obstacle boxes
    :return: True if in collision, False if safe
    """
    fk = FK()
    joint_positions, _ = fk.forward(q)

    # Define parameters for collision shapes
    link_radius = 0.07  # Example radius for cylindrical links
    joint_radius = 0.07  # Example radius for spherical joints

    # Check collisions for each joint (treated as spheres)
    for i in range(8):  # 8 positions including end-effector
        pt = joint_positions[i]
        for obstacle in obstacles:
            if detectCollision(pt.reshape(1, 3), pt.reshape(1, 3), obstacle)[0]:
                return True
            
            # Additionally check if the distance from the joint center to the obstacle is within the joint radius
            distance_to_obstacle = detectDistanceToBox(pt, obstacle)
            if distance_to_obstacle <= joint_radius:
                return True  # Collision detected due to proximity of the joint

    # Check collisions for each link (considering their radius)
    for i in range(7):  # 7 links between 8 joint positions
        pt1 = joint_positions[i]
        pt2 = joint_positions[i + 1]

        # Calculate the vector from pt1 to pt2
        link_vector = pt2 - pt1
        link_length = np.linalg.norm(link_vector)

        # Normalize the vector to get the direction
        if link_length > 0:
            direction = link_vector / link_length

            # Check for collision with each obstacle
            for obstacle in obstacles:
                # Calculate distance from obstacle to line segment (pt1 to pt2)
                distance_to_obstacle = detectDistanceToBox(pt1, obstacle)

                # If this distance is less than or equal to the link radius, there is a collision
                if distance_to_obstacle <= link_radius:
                    return True

    return False

def detectDistanceToBox(point, box):
    """
    Calculate minimum distance from a point to a box.
    :param point: The point (3D) to check against.
    :param box: [xmin, ymin, zmin, xmax, ymax, zmax] defining the box.
    :return: Minimum distance from point to box.
    """
    
    # Clamp point coordinates within box dimensions
    clamped_x = np.clip(point[0], box[0], box[3])
    clamped_y = np.clip(point[1], box[1], box[4])
    clamped_z = np.clip(point[2], box[2], box[5])

    # Calculate squared distance from point to clamped point (closest point on box)
    closest_point = np.array([clamped_x, clamped_y, clamped_z])
    
    return np.linalg.norm(point - closest_point)

def pointToLineSegmentDistance(point, line_start, direction, length):
    """
    Calculate minimum distance from a point to a line segment.
    :param point: The point (3D) to check against.
    :param line_start: Starting point of the line segment.
    :param direction: Direction vector of the line segment (normalized).
    :param length: Length of the line segment.
    :return: Minimum distance from point to line segment.
    """

    # Vector from start point to point
    start_to_point = point - line_start

    # Project start_to_point onto direction vector
    projection_length = np.dot(start_to_point, direction)

    # Clamp projection length to be within [0, length]
    projection_length = np.clip(projection_length, 0, length)

    # Find closest point on segment
    closest_point_on_segment = line_start + projection_length * direction

    # Calculate distance from closest point on segment to point
    return np.linalg.norm(point - closest_point_on_segment)

def sample_random_config(lowerLim, upperLim):
    """Sample a random configuration within joint limits."""
    return np.random.uniform(lowerLim, upperLim, size=7)

def nearest_neighbor(tree, sample):
    """Find the index of the nearest neighbor in the tree."""
    tree_array = np.array(tree)
    distances = np.linalg.norm(tree_array - sample, axis=1)
    return np.argmin(distances)  # Return the index of the nearest neighbor

# Helper function to interpolate between configurations
def interpolateConfig(q1, q2, step_size=0.45):
    """
    Interpolate between two configurations with given step size
    """
    dist = np.linalg.norm(q2 - q1)
    if dist < step_size:
        return q2
    else:
        return q1 + step_size * (q2 - q1) / dist

def check_connection(config_new, tree):
    """Check if config_new can connect to any node in tree."""

    for node in tree:
        if np.linalg.norm(config_new - node) < 0.01:  # Adjust threshold as necessary
            return True

    return False

def reconstruct_path(tree, goal_idx):
    """Reconstruct path from start to goal."""
    path = [tree[goal_idx]]
    current_idx = goal_idx

    while current_idx > 0:
        current_idx = nearest_neighbor(tree[:current_idx], tree[current_idx])
        path.append(tree[current_idx])

    return path[::-1]  # Reverse to get path from start to goal

# Main RRT implementation
def rrt(map, start, goal):
    """
    Implement RRT algorithm for path planning.
    :param map: the map struct containing obstacles.
    :param start: start pose of the robot (1x7).
    :param goal: goal pose of the robot (1x7).
    :return: returns an mx7 matrix, where each row consists of the configuration of
             the Panda at a point on the path. The first row is start and
             the last row is goal. If no path is found, PATH is empty.
    """

    # RRT parameters
    T_start = [start]
    T_goal = [goal]
    n_iter = 10000
    lowerLim = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    upperLim = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    step_size = 0.5
    goal_sample_rate = 0.3

    for _ in range(n_iter):
        # Sample a random configuration in Qfree
        if random.random() < goal_sample_rate:
            q_random = goal
        else:
            q_random = sample_random_config(lowerLim, upperLim)
        #print("Q random:", q_random)

        # In the T_start extension
        q_nearest_start_idx = nearest_neighbor(T_start, q_random)
        q_nearest_start = T_start[q_nearest_start_idx]

        # In the T_goal extension
        q_nearest_goal_idx = nearest_neighbor(T_goal, q_random)
        q_nearest_goal = T_goal[q_nearest_goal_idx]

        q_start_new = interpolateConfig(q_nearest_start, q_random)
        # When checking connection to T_goal
        q_end_new = interpolateConfig(q_nearest_goal, q_random)

        if not isRobotCollided(q_start_new, map.obstacles):
            T_start.append(q_start_new)
            #print(f"Added to T_start: {q_start_new}")

            if check_connection(q_start_new, T_goal):
                print("Connection found between T_start and T_goal!")
                start_path = reconstruct_path(T_start, len(T_start) - 1)
                goal_idx = nearest_neighbor(T_goal, q_start_new)
                goal_path = reconstruct_path(T_goal, goal_idx)
                full_path = np.array(start_path + [q_start_new] + goal_path[::-1])
                return np.array(full_path)  # Ensure start and goal are included

        if not isRobotCollided(q_end_new, map.obstacles):
            T_goal.append(q_end_new)
            #print(f"Added to T_goal: {q_end_new}")

            if check_connection(q_end_new, T_start):
                print("Connection found between T_goal and T_start!")
                # Find the index of the nearest node in T_start to q_end_new
                start_nearest_idx = nearest_neighbor(T_start, q_end_new)
                start_path = reconstruct_path(T_start, start_nearest_idx)
                goal_path = reconstruct_path(T_goal, len(T_goal) - 1)
                full_path = start_path + [q_end_new] + goal_path[::-1]
                return np.array(full_path)

    return np.array([]).reshape(0, 7)  # No path found

if __name__ == '__main__':
    map_struct = loadmap("../maps/map1.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, -1 , 1.57, -2.07, 0, -1.0, 0.7])
    # goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    path = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal))

    if len(path)> 0:
        print("Path found:")
        print(path)
    else:
        print("No path found.")