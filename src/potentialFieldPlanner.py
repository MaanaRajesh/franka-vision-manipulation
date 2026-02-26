
import sys
sys.path.append("/home/student/meam520_ws/src/meam520_labs")
import numpy as np
from math import pi, acos
from scipy.linalg import null_space
from copy import deepcopy
from lib.calculateFKJac import FK_Jac
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap
from time import perf_counter


class PotentialFieldPlanner:

    # JOINT LIMITS
    lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    center = lower + (upper - lower) / 2 # compute middle of range of motion of each joint
    fk = FK_Jac()

    def __init__(self, tol=1e-4, max_steps=500, min_step_size=1e-5):
        """
        Constructs a potential field planner with solver parameters.

        PARAMETERS:
        tol - the maximum distance between two joint sets
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
        optimizer has converged
        """

        # YOU MAY NEED TO CHANGE THESE PARAMETERS

        # solver parameters
        self.tol = tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size


    ######################
    ## Helper Functions ##
    ######################
    # The following functions are provided to you to help you to better structure your code
    # You don't necessarily have to use them. You can also edit them to fit your own situation 

    @staticmethod
    def attractive_force(target, current, attract_strength, mode = 2):
        """
        Helper function for computing the attactive force between the current position and
        the target position for one joint. Computes the attractive force vector between the 
        target joint position and the current joint position 

        INPUTS:
        target - 3x1 numpy array representing the desired joint position in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame

        OUTPUTS:
        att_f - 3x1 numpy array representing the force vector that pulls the joint 
        from the current position to the target position 
        """

        ## STUDENT CODE STARTS HERE

        att_f = np.zeros((3, 1))
        displacement = current - target
        d_threshold = 0.12
        # attract_strength = 14
        if np.linalg.norm(displacement) < 1e-5:
            return np.zeros((3, 1))
        elif np.linalg.norm(displacement) ** 2 < d_threshold:
            att_f = -attract_strength * displacement
        elif mode == 2:
            att_f = -displacement / np.linalg.norm(displacement)
        elif mode == 1:
            att_f = -attract_strength * displacement
        ## END STUDENT CODE

        return att_f

    @staticmethod
    def repulsive_force(obstacle, current, repulse_strength, repulse_distance, unitvec=np.zeros((3,1))):
        """
        Helper function for computing the repulsive force between the current position
        of one joint and one obstacle. Computes the repulsive force vector between the 
        obstacle and the current joint position 

        INPUTS:
        obstacle - 1x6 numpy array representing the an obstacle box in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame
        unitvec - 3x1 numpy array representing the unit vector from the current joint position 
        to the closest point on the obstacle box 

        OUTPUTS:
        rep_f - 3x1 numpy array representing the force vector that pushes the joint 
        from the obstacle
        """

        ## STUDENT CODE STARTS HERE

        rep_f = np.zeros((3, 1))
        distance, unit = PotentialFieldPlanner.dist_point2box(current[np.newaxis, :], obstacle[0])
        # rou_0 = 0.05
        # repulse_strength = 0.25
        if repulse_distance == 0.12:
            box = obstacle[0]
            p = np.array(current).flatten()
            boxMin = np.array([box[0], box[1], box[2]])
            boxMax = np.array([box[3], box[4], box[5]])
            dx = np.maximum(0, np.maximum(boxMin[0] - p[0], p[0] - boxMax[0]))
            dy = np.maximum(0, np.maximum(boxMin[1] - p[1], p[1] - boxMax[1]))
            dz = np.maximum(0, np.maximum(boxMin[2] - p[2], p[2] - boxMax[2]))
            dist = np.array([dx, dy, dz])
            unitvec = dist / (np.linalg.norm(dist) + 1e-8)
        if distance > repulse_distance:
            magnitude = 0
        elif distance > 0:
            if repulse_distance == 0.12:
                magnitude = -repulse_strength * (1/distance - 1/repulse_distance) * (1/distance ** 2)
            else:
                magnitude = repulse_strength * (1/distance - 1/repulse_distance) * (1/distance ** 2)
        else:
            return np.zeros((3, 1))
        rep_f = magnitude * unitvec

        ## END STUDENT CODE

        return rep_f

    @staticmethod
    def dist_point2box(p, box):
        """
        Helper function for the computation of repulsive forces. Computes the closest point
        on the box to a given point 
    
        INPUTS:
        p - nx3 numpy array of points [x,y,z]
        box - 1x6 numpy array of minimum and maximum points of box

        OUTPUTS:
        dist - nx1 numpy array of distance between the points and the box
                dist > 0 point outside
                dist = 0 point is on or inside box
        unit - nx3 numpy array where each row is the corresponding unit vector 
        from the point to the closest spot on the box
            norm(unit) = 1 point is outside the box
            norm(unit)= 0 point is on/inside the box

         Method from MultiRRomero
         @ https://stackoverflow.com/questions/5254838/
         calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
        """
        # THIS FUNCTION HAS BEEN FULLY IMPLEMENTED FOR YOU

        # Get box info
        boxMin = np.array([box[0], box[1], box[2]])
        boxMax = np.array([box[3], box[4], box[5]])
        boxCenter = boxMin*0.5 + boxMax*0.5
        p = np.array(p)

        # Get distance info from point to box boundary
        dx = np.amax(np.vstack([boxMin[0] - p[:, 0], p[:, 0] - boxMax[0], np.zeros(p[:, 0].shape)]).T, 1)
        dy = np.amax(np.vstack([boxMin[1] - p[:, 1], p[:, 1] - boxMax[1], np.zeros(p[:, 1].shape)]).T, 1)
        dz = np.amax(np.vstack([boxMin[2] - p[:, 2], p[:, 2] - boxMax[2], np.zeros(p[:, 2].shape)]).T, 1)

        # convert to distance
        distances = np.vstack([dx, dy, dz]).T
        dist = np.linalg.norm(distances, axis=1)

        # Figure out the signs
        signs = np.sign(boxCenter-p)

        # Calculate unit vector and replace with
        unit = distances / dist[:, np.newaxis] * signs
        unit[np.isnan(unit)] = 0
        unit[np.isinf(unit)] = 0
        return dist, unit

    @staticmethod
    def compute_forces(target, obstacle, current, attract_para = None, repulse_para = None, repulse_distance = None):
        """
        Helper function for the computation of forces on every joints. Computes the sum 
        of forces (attactive, repulsive) on each joint. 

        INPUTS:
        target - 3x9 numpy array representing the desired joint/end effector positions 
        in the world frame
        obstacle - nx6 numpy array representing the obstacle box min and max positions
        in the world frame
        current- 3x9 numpy array representing the current joint/end effector positions 
        in the world frame

        OUTPUTS:
        joint_forces - 3x9 numpy array representing the force vectors on each 
        joint/end effector
        """

        ## STUDENT CODE STARTS HERE

        joint_forces = np.zeros((3, 9))
        if attract_para is None:
            attract_para = np.array([14, 15, 21, 14, 14, 14, 14, 14, 14])
            mode = 1
        else:
            mode = 2
        if repulse_para is None:
            repulse_para = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
        if repulse_distance is None:
            repulse_distance = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        flag = True
        try:
            if not obstacle[0]:
                flag = False
        except:
            flag = True
        if flag:
            for ob in obstacle:
                for i in range(9):
                    closest_point = np.clip(current[:, i], ob[0, :3], ob[0, 3:])
                    unitvec = (current[:, i] - closest_point).reshape(3, 1)
                    att_force = PotentialFieldPlanner.attractive_force(target[:,i], current[:,i], attract_para[i], mode)
                    repulse_force = PotentialFieldPlanner.repulsive_force(ob, current[:,i], repulse_para[i], repulse_distance[i], unitvec)
                    joint_forces[:, i] = joint_forces[:, i] + att_force.flatten() + repulse_force.flatten()
        else:
            for i in range(9):
                att_force = PotentialFieldPlanner.attractive_force(target[:,i], current[:,i], attract_para[i])
                joint_forces[:, i] = joint_forces[:, i] + att_force.flatten()

        ## END STUDENT CODE

        return joint_forces
    
    @staticmethod
    def compute_torques(joint_forces, q):
        """
        Helper function for converting joint forces to joint torques. Computes the sum 
        of torques on each joint.

        INPUTS:
        joint_forces - 3x9 numpy array representing the force vectors on each 
        joint/end effector
        q - 1x7 numpy array representing the current joint angles

        OUTPUTS:
        joint_torques - 1x9 numpy array representing the torques on each joint 
        """

        ## STUDENT CODE STARTS HERE

        joint_torques = np.zeros((1, 9))
        for i in range(1, 10):
            jacobian = PotentialFieldPlanner.fk.calcJacobian(q, i)
            joint_torques += jacobian[:3, :].T @ joint_forces[:, i-1]
        
        joint_torques = joint_torques[np.newaxis, :]

        ## END STUDENT CODE

        return joint_torques

    @staticmethod
    def q_distance(target, current):
        """
        Helper function which computes the distance between any two
        vectors.

        This data can be used to decide whether two joint sets can be
        considered equal within a certain tolerance.

        INPUTS:
        target - 1x7 numpy array representing some joint angles
        current - 1x7 numpy array representing some joint angles

        OUTPUTS:
        distance - the distance between the target and the current joint sets 

        """

        ## STUDENT CODE STARTS HERE

        distance = np.linalg.norm(target - current)

        ## END STUDENT CODE

        return distance
    
    @staticmethod
    def compute_gradient(q, target, map_struct, attract_para = None, repulse_para = None, repulse_dist = None):
        """
        Computes the joint gradient step to move the current joint positions to the
        next set of joint positions which leads to a closer configuration to the goal 
        configuration 

        INPUTS:
        q - 1x7 numpy array. the current joint configuration, a "best guess" so far for the final answer
        target - 1x7 numpy array containing the desired joint angles
        map_struct - a map struct containing the obstacle box min and max positions

        OUTPUTS:
        dq - 1x7 numpy array. a desired joint velocity to perform this task. 
        """

        ## STUDENT CODE STARTS HERE

        dq = np.zeros((1, 7))
        current_pos,_ = PotentialFieldPlanner.fk.forward_expanded(q)
        target_pos,_ = PotentialFieldPlanner.fk.forward_expanded(target)
        joint_forces = PotentialFieldPlanner.compute_forces(target_pos[1:, :].T, map_struct, current_pos[1:, :].T, attract_para, repulse_para, repulse_dist)
        joint_torques = PotentialFieldPlanner.compute_torques(joint_forces, q)

        dq = joint_torques[0, :, :7] / np.linalg.norm(joint_torques[0, :, :7])
        # dq = joint_torques[0, :, :7]

        ## END STUDENT CODE

        return dq

    ###############################
    ### Potential Feild Solver  ###
    ###############################

    def plan(self, map_struct, start, goal):
        """
        Uses potential field to move the Panda robot arm from the startng configuration to
        the goal configuration.

        INPUTS:
        map_struct - a map struct containing min and max positions of obstacle boxes 
        start - 1x7 numpy array representing the starting joint angles for a configuration 
        goal - 1x7 numpy array representing the desired joint angles for a configuration

        OUTPUTS:
        q - nx7 numpy array of joint angles [q0, q1, q2, q3, q4, q5, q6]. This should contain
        all the joint angles throughout the path of the planner. The first row of q should be
        the starting joint angles and the last row of q should be the goal joint angles. 
        """

        q_path = np.array([]).reshape(0,7)
        q_path = np.vstack([q_path, start])
        current_q = start
        max_iterations = 1000
        tolerance = 0.019
        step_size = 1
        collided = True
        min_joint_limit = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        max_joint_limit = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        previous_error = 0
        # error_list = []

        while True:

            ## STUDENT CODE STARTS HERE
            
            # The following comments are hints to help you to implement the planner
            # You don't necessarily have to follow these steps to complete your code 
            
            # Compute gradient 
            # TODO: this is how to change your joint angles
            # if len(q_path) > 30 and (q_path[-1]-q_path[-3])[0] == 0:
            #     step_size = step_size / 2
            # if q_path.shape[0] > 500:
            #     step_size = 0.025
            # if q_path.shape[0] > 550:
            #     step_size = 0.001
            # if q_path.shape[0] > 700:
            #     step_size = 0.0001

            # current error < previous error
            # alpha min(min lim max lim)(0.01 and 1)
            # else
            # alpha max
            dq = PotentialFieldPlanner.compute_gradient(current_q, goal, map_struct)
            dq = dq * step_size
            next_q = current_q + dq

            next_q = np.clip(next_q, min_joint_limit, max_joint_limit)

            # Termination Conditions
            if self.q_distance(goal, next_q) < tolerance or len(q_path) - 1 > max_iterations: # TODO: check termination conditions
                break # exit the while loop if conditions are met!

            # YOU NEED TO CHECK FOR COLLISIONS WITH OBSTACLES
            # TODO: Figure out how to use the provided function
            flag = True
            try:
                if not map_struct[0]:
                    flag = False
            except:
                flag = True
            while collided == True and flag:
                current_pos,_ = PotentialFieldPlanner.fk.forward_expanded(current_q)
                next_pos,_ = PotentialFieldPlanner.fk.forward_expanded(next_q.flatten())
                obstacle_box = [box[0] for box in map_struct]
                collided = detectCollision(current_pos[1:], next_pos[1:], obstacle_box[0])
                collided = any(collided)
                if collided == True:
                    random_step = (np.random.rand(7) - 0.5) * 2 * 0.05
                    next_q = current_q + random_step
                    next_q = np.clip(next_q, min_joint_limit, max_joint_limit)
                    # joint_num = int(np.random.random_sample(1) * 7)
                    # sample_val = (np.random.rand(1) - 0.5) * 2 * 0.1
                    # next_q[0, joint_num-1] = sample_val
            collided = True
            # while dist == 0:
            #     next_pos = PotentialFieldPlanner.fk.forward_expanded(next_q)
            #     for obstacle in map_struct:
            #         dist,_ = PotentialFieldPlanner.dist_point2box(next_pos, obstacle)
            #         if dist == 0:
            #             random_step = (np.random.rand(7) - 0.5) * 2 * step_size
            #             next_q = current_q + random_step

            # YOU MAY NEED TO DEAL WITH LOCAL MINIMA HERE
            # TODO: when detect a local minima, implement a random walk
            if np.linalg.norm(dq/step_size) < 1e-1 and self.q_distance(goal, next_q) > 0.1:
                # Form a new joint angle instead of using random walk
                # joint_num = int(np.random.random_sample(1) * 7)
                # sample_val = (np.random.rand(1) - 0.5) * 2 * 0.1
                # next_q[0, joint_num-1] += sample_val
                random_step = (np.random.rand(7) - 0.5) * 2 * 0.05
                next_q = current_q + random_step
                print("Local Minimum detected.")
            
            current_q = next_q.flatten()
            current_error = PotentialFieldPlanner.q_distance(current_q, goal)
            if current_error < 0.1:
                step_size = np.random.random_sample(1) * 0.0005 + 0.0005
            elif current_error < 1:
                step_size = np.random.random_sample(1) * 0.005 + 0.005
            elif current_error < 2:
                step_size = np.random.random_sample(1) * 0.02 + 0.02
            elif current_error < 5:
                step_size = np.random.random_sample(1) * 0.05 + 0.05
            else:
                step_size = np.random.random_sample(1) * 0.5 + 0.5
            # if current_error < previous_error and previous_error != 0:
            #     step_size -= 0.0001
            # elif current_error > previous_error and previous_error != 0:
            #     step_size += 0.0001
            # error_list.append(current_error)
            previous_error = current_error
            q_path = np.vstack([q_path, current_q])
        
        if current_error > 0.02:
            global start_time
            start_time = perf_counter()
            # Planner 2
            trajectory = np.array([start])  # Initialize trajectory with the start configuration
            current_q = start
            step_size = 0.5  # Initial step magnitude
            previous_error = np.inf
            stalled_steps = 0  # Counter to track steps with no progress

            for iteration in range(1000):
                current_q = current_q.flatten()
                # Compute joint adjustment with scaled forces
                delta_q = self.compute_gradient(current_q, goal, map_struct, attract_para=[30,30,30,30,30,30,30,15,15], repulse_para=[0.000] * 9, repulse_dist=[0.12] * 9)
                proposed_q = current_q + step_size * delta_q  # Update joint configuration along gradient
                proposed_q = np.clip(proposed_q, min_joint_limit, max_joint_limit)  # Apply joint limits

                # Measure improvement in error towards goal
                new_error = self.q_distance(proposed_q, goal)

                # Accept proposed configuration if it reduces error, and decrease step size
                if new_error < 0.1:
                    step_size = np.random.random_sample(1) * 0.0005 + 0.0005
                elif new_error < 1:
                    step_size = np.random.random_sample(1) * 0.05 + 0.005
                elif new_error < 2:
                    step_size = np.random.random_sample(1) * 0.05 + 0.05
                elif new_error < 5:
                    step_size = np.random.random_sample(1) * 0.05 + 0.05
                else:
                    step_size = np.random.random_sample(1) * 0.5 + 0.5

                if new_error < previous_error:
                    # current_q = proposed_q
                    # previous_error = new_error
                    # step_size = max(0.01, step_size * 0.95)  # Gradually reduce step magnitude
                    # trajectory = np.vstack((trajectory, current_q.copy()))  # Append to trajectory
                    stalled_steps = 0  # Reset stalled steps count
                
                else:
                    # Increment no-progress counter if no improvement is made
                    stalled_steps += 1
                
                current_q = proposed_q
                previous_error = new_error

                trajectory = np.vstack((trajectory, proposed_q.copy()))  # Append to trajectory

                # Check if within acceptable proximity to the goal
                if new_error < 0.019:
                    break  # Terminate if within tolerance threshold

                joint_positions, _ = self.fk.forward_expanded(current_q.flatten())
                joints_to_check = np.concatenate((joint_positions[:7], joint_positions[8:]), axis=0)
                for obstacle in map_struct.obstacles:
                    if any(detectCollision(joints_to_check[:-1], joints_to_check[1:], obstacle)):
                        stalled_steps += 1
                        break

                # If stalled, perturb the configuration slightly to escape local minima
                if stalled_steps >= 10 and new_error > 0.02:
                    current_q[-4:] += np.random.uniform(-0.05, 0.05, size=current_q[-4:].shape)  # Randomize last 4 joints
                    current_q = np.clip(current_q, min_joint_limit, max_joint_limit)  # Apply joint limits post-perturbation
                    stalled_steps = 0  # Reset stalled steps after perturbation
            
        if current_error < 0.02:
            return q_path
        
        elif new_error < 0.02:
            return trajectory
        
        else:
            return trajectory
        

        ## END STUDENT CODE

################################
## Simple Testing Environment ##
################################

if __name__ == "__main__":

    np.set_printoptions(suppress=True,precision=5)

    planner = PotentialFieldPlanner()
    
    # inputs 
    map_struct = loadmap("../maps/map2.txt")
    start = np.array([0, -1, 0, -2, 0, 2, 0])
    goal = np.array([-1.2, 0.4 , -1.57, -2.07, -1.0, 3, 0.7])
    
    # potential field planning
    start_time = perf_counter()
    q_path = planner.plan(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    stop_time = perf_counter()
    time = stop_time - start_time
    distance = 0
    
    # show results
    for i in range(q_path.shape[0]):
        if i == 0:
            past_point, _ = PotentialFieldPlanner.fk.forward_expanded(start.flatten())
        else:
            past_point, _ = PotentialFieldPlanner.fk.forward_expanded(q_path[i-1, :].flatten())
        current_point, _ = PotentialFieldPlanner.fk.forward_expanded(q_path[i, :].flatten())
        print(current_point[-3].reshape(3) - past_point[-3].reshape(3))
        distance += np.linalg.norm(current_point[-3].reshape(3) - past_point[-3].reshape(3))
        # distance += PotentialFieldPlanner.q_distance(q_path[i, :], q_path[i-1, :])
        error = PotentialFieldPlanner.q_distance(q_path[i, :], goal)
        print('iteration:',i,' q =', q_path[i, :], ' error={error}'.format(error=error))

    print("q path: ", q_path)
    print("Time: ", time)
    print("Distance: ", distance)