import sys
sys.path.append("/home/student/meam520_ws/src/meam520_labs")
import numpy as np
from math import pi, acos
from scipy.linalg import null_space

from lib.calcJacobian import calcJacobian
from lib.calculateFK import FK
from lib.calcAngDiff import calcAngDiff
# from lib.IK_velocity import IK_velocity  #optional


class IK:

    # JOINT LIMITS
    lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    center = lower + (upper - lower) / 2 # compute middle of range of motion of each joint
    fk = FK()

    def __init__(self,linear_tol=1e-4, angular_tol=1e-3, max_steps=1000, min_step_size=1e-5):
        """
        Constructs an optimization-based IK solver with given solver parameters.
        Default parameters are tuned to reasonable values.

        PARAMETERS:
        linear_tol - the maximum distance in meters between the target end
        effector origin and actual end effector origin for a solution to be
        considered successful
        angular_tol - the maximum angle of rotation in radians between the target
        end effector frame and actual end effector frame for a solution to be
        considered successful
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
        optimizer has converged
        """

        # solver parameters
        self.linear_tol = linear_tol
        self.angular_tol = angular_tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size


    ######################
    ## Helper Functions ##
    ######################

    @staticmethod
    def displacement_and_axis(target, current):
        """
        Helper function for the End Effector Task. Computes the displacement
        vector and axis of rotation from the current frame to the target frame

        This data can also be interpreted as an end effector velocity which will
        bring the end effector closer to the target position and orientation.

        INPUTS:
        target - 4x4 numpy array representing the desired transformation from
        end effector to world
        current - 4x4 numpy array representing the "current" end effector orientation

        OUTPUTS:
        displacement - a 3-element numpy array containing the displacement from
        the current frame to the target frame, expressed in the world frame
        axis - a 3-element numpy array containing the axis of the rotation from
        the current frame to the end effector frame. The magnitude of this vector
        must be sin(angle), where angle is the angle of rotation around this axis
        """

        ## STUDENT CODE STARTS HERE
        target_pos = target[:3, 3]
        target_ori = target[:3, :3]
        current_pos = current[:3, 3]
        current_ori = current[:3, :3]
        displacement = target_pos - current_pos

        axis = calcAngDiff(target_ori, current_ori)

        ## END STUDENT CODE
        return displacement, axis

    @staticmethod
    def distance_and_angle(G, H):
        """
        Helper function which computes the distance and angle between any two
        transforms.

        This data can be used to decide whether two transforms can be
        considered equal within a certain linear and angular tolerance.

        Be careful! Using the axis output of displacement_and_axis to compute
        the angle will result in incorrect results when |angle| > pi/2

        INPUTS:
        G - a 4x4 numpy array representing some homogenous transformation
        H - a 4x4 numpy array representing some homogenous transformation

        OUTPUTS:
        distance - the distance in meters between the origins of G & H
        angle - the angle in radians between the orientations of G & H
        """
        
        ## STUDENT CODE STARTS HERE
        distance = np.linalg.norm(G[:3, 3] - H[:3, 3])
        R1 = G[:3, :3]
        R2 = H[:3, :3]
        R_rel = np.dot(R2.T, R1)
        cos_value = (np.trace(R_rel) - 1)/2
        cos_value = np.clip(cos_value, -1.0, 1.0)
        angle = np.arccos(cos_value)

        ## END STUDENT CODE
        return distance, angle

    def is_valid_solution(self,q,target):
        """
        Given a candidate solution, determine if it achieves the primary task
        and also respects the joint limits.

        INPUTS
        q - the candidate solution, namely the joint angles
        target - 4x4 numpy array representing the desired transformation from
        end effector to world

        OUTPUTS:
        success - a Boolean which is True if and only if the candidate solution
        produces an end effector pose which is within the given linear and
        angular tolerances of the target pose, and also respects the joint
        limits.
        """

        ## STUDENT CODE STARTS HERE
        success = False
        message = ""

        # Check joint limits
        within_limits = np.all((q >= IK.lower) & (q <= IK.upper))
        if not within_limits:
            message = "Joint limits violated."
            return success, message  # Immediately return if joint limits are not met

        # Compute achieved end-effector transformation from forward kinematics
        achieved = IK.fk.forward(q)[1]

        # Calculate distance and angle using distance_and_angle helper
        distance, angle = IK.distance_and_angle(achieved, target)

        # Check distance within tolerance
        if distance > self.linear_tol:
            message = f"Position tolerance exceeded. Distance: {distance:.4f}, Tolerance: {self.linear_tol}"
            return success, message

        # Check angle within tolerance
        if angle > self.angular_tol:
            message = f"Orientation tolerance exceeded. Angle: {angle:.4f}, Tolerance: {self.angular_tol}"
            return success, message

        # If all checks passed
        success = True
        message = "Valid solution found within tolerances and joint limits."

        ## END STUDENT CODE
        return success, message

    ####################
    ## Task Functions ##
    ####################

    @staticmethod
    def end_effector_task(q,target,method):
        """
        Primary task for IK solver. Computes a joint velocity which will reduce
        the error between the target end effector pose and the current end
        effector pose (corresponding to configuration q).

        INPUTS:
        q - the current joint configuration, a "best guess" so far for the final answer
        target - a 4x4 numpy array containing the desired end effector pose
        method - a boolean variable that determines to use either 'J_pseudo' or 'J_trans' 
        (J pseudo-inverse or J transpose) in your algorithm
        
        OUTPUTS:
        dq - a desired joint velocity to perform this task, which will smoothly
        decay to zero magnitude as the task is achieved
        """

        ## STUDENT CODE STARTS HERE
        dq = np.zeros(7)
        current = IK.fk.forward(q)[1]
        displacement, axis = IK.displacement_and_axis(target, current)
        desired_velocity = np.hstack([displacement, axis])[:, np.newaxis]
        J = calcJacobian(q)
        if method == "J_pseudo":
            J_pinv = np.linalg.pinv(J)
            dq = (J_pinv @ desired_velocity).flatten()

        if method == "J_trans":
            dq = (J.T @ desired_velocity).flatten()

        ## END STUDENT CODE
        return dq

    @staticmethod
    def joint_centering_task(q,rate=5e-1): 
        """
        Secondary task for IK solver. Computes a joint velocity which will
        reduce the offset between each joint's angle and the center of its range
        of motion. This secondary task acts as a "soft constraint" which
        encourages the solver to choose solutions within the allowed range of
        motion for the joints.

        INPUTS:
        q - the joint angles
        rate - a tunable parameter dictating how quickly to try to center the
        joints. Turning this parameter improves convergence behavior for the
        primary task, but also requires more solver iterations.

        OUTPUTS:
        dq - a desired joint velocity to perform this task, which will smoothly
        decay to zero magnitude as the task is achieved
        """

        # normalize the offsets of all joints to range from -1 to 1 within the allowed range
        offset = 2 * (q - IK.center) / (IK.upper - IK.lower)
        dq = rate * -offset # proportional term (implied quadratic cost)

        return dq
        
    ###############################
    ## Inverse Kinematics Solver ##
    ###############################

    def inverse(self, target, seed, method, alpha):
        """
        Uses gradient descent to solve the full inverse kinematics of the Panda robot.

        INPUTS:
        target - 4x4 numpy array representing the desired transformation from
        end effector to world
        seed - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6], which
        is the "initial guess" from which to proceed with optimization
        method - a boolean variable that determines to use either 'J_pseudo' or 'J_trans' 
        (J pseudo-inverse or J transpose) in your algorithm

        OUTPUTS:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6], giving the
        solution if success is True or the closest guess if success is False.
        success - True if the IK algorithm successfully found a configuration
        which achieves the target within the given tolerance. Otherwise False
        rollout - a list containing the guess for q at each iteration of the algorithm
        """

        q = seed
        rollout = []

        ## STUDENT CODE STARTS HERE

        
        ## gradient descent:
        while True:
            rollout.append(q)

            # Primary Task - Achieve End Effector Pose
            dq_ik = IK.end_effector_task(q,target, method)

            # Secondary Task - Center Joints
            dq_center = IK.joint_centering_task(q)

            J = calcJacobian(q)

            ## Task Prioritization
            if method == "J_pseudo":
                J_pinv = np.linalg.pinv(J)
                null_space_projection = np.eye(7) - J_pinv @ J 
                dq_center = null_space_projection @ dq_center
            
            if method == "J_trans":
                J_trans = J.T
                null_space_projection = np.eye(7) - J.T @ np.linalg.inv(J @ J.T) @ J
                dq_center = null_space_projection @ dq_center
                
            dq = dq_ik + dq_center
            # Check termination conditions
            if len(rollout) >= self.max_steps:
                break
            elif np.linalg.norm(dq) < self.min_step_size:
                break

            # update q
            q = q + alpha * dq

        ## END STUDENT CODE

        success, message = self.is_valid_solution(q,target)
        return q, rollout, success, message

################################
## Simple Testing Environment ##
################################

if __name__ == "__main__":

    np.set_printoptions(suppress=True,precision=5)

    ik = IK()

    # matches figure in the handout
    # seed = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
    initial_guess = np.array([ 0.94324187,  0.31310008 , 0.29354613 ,-1.22159909, -0.08934577 , 1.52230772,  2.00432444])

    # target = np.array([
    #     [0,-1,0,-0.2],
    #     [-1,0,0,0],
    #     [0,0,-1,.5],
    #     [0,0,0, 1],
    # ])
    target_detect = np.array([
            [0,1,0,0],
            [1,0,0,-0.295],
            [0,0,-1,0.4],
            [0,0,0,1],])
    T_b_w = np.array([[1,0,0,0],\
                    [0,1,0,0.990],\
                    [0,0,1,0],\
                    [0,0,0,1]])
    target_detect = T_b_w @ target_detect

    # Using pseudo-inverse 
    q_pseudo, rollout_pseudo, success_pseudo, message_pseudo = ik.inverse(target_detect, initial_guess, method='J_pseudo', alpha=.5)

    for i, q_pseudo in enumerate(rollout_pseudo):
        joints, pose = ik.fk.forward(q_pseudo)
        d, ang = IK.distance_and_angle(target_detect,pose)
        print('iteration:',i,' q =',q_pseudo, ' d={d:3.4f}  ang={ang:3.3f}'.format(d=d,ang=ang))

    # Using pseudo-inverse 
    q_trans, rollout_trans, success_trans, message_trans = ik.inverse(target_detect, initial_guess, method='J_trans', alpha=.5)

    for i, q_trans in enumerate(rollout_trans):
        joints, pose = ik.fk.forward(q_trans)
        d, ang = IK.distance_and_angle(target_detect,pose)
        print('iteration:',i,' q =',q_trans, ' d={d:3.4f}  ang={ang:3.3f}'.format(d=d,ang=ang))

    # compare
    print("\n method: J_pseudo-inverse")
    print("   Success: ",success_pseudo, ":  ", message_pseudo)
    print("   Solution: ",q_pseudo)
    print("   #Iterations : ", len(rollout_pseudo))
    print("\n method: J_transpose")
    print("   Success: ",success_trans, ":  ", message_trans)
    print("   Solution: ",q_trans)
    print("   #Iterations :", len(rollout_trans),'\n')




# import sys
# sys.path.append("/home/student/meam520_ws/src/meam520_labs")
# import numpy as np
# from math import pi, acos
# from scipy.linalg import null_space

# from lib.calcJacobian import calcJacobian
# from lib.calculateFK import FK
# from lib.calcAngDiff import calcAngDiff
# # from lib.IK_velocity import IK_velocity  #optional


# class IK:

#     # JOINT LIMITS
#     lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
#     upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

#     center = lower + (upper - lower) / 2 # compute middle of range of motion of each joint
#     fk = FK()

#     def __init__(self,linear_tol=1e-4, angular_tol=1e-3, max_steps=1000, min_step_size=1e-5):
#         """
#         Constructs an optimization-based IK solver with given solver parameters.
#         Default parameters are tuned to reasonable values.

#         PARAMETERS:
#         linear_tol - the maximum distance in meters between the target end
#         effector origin and actual end effector origin for a solution to be
#         considered successful
#         angular_tol - the maximum angle of rotation in radians between the target
#         end effector frame and actual end effector frame for a solution to be
#         considered successful
#         max_steps - number of iterations before the algorithm must terminate
#         min_step_size - the minimum step size before concluding that the
#         optimizer has converged
#         """

#         # solver parameters
#         self.linear_tol = linear_tol
#         self.angular_tol = angular_tol
#         self.max_steps = max_steps
#         self.min_step_size = min_step_size


#     ######################
#     ## Helper Functions ##
#     ######################

#     @staticmethod
#     def displacement_and_axis(target, current):
#         """
#         Helper function for the End Effector Task. Computes the displacement
#         vector and axis of rotation from the current frame to the target frame

#         This data can also be interpreted as an end effector velocity which will
#         bring the end effector closer to the target position and orientation.

#         INPUTS:
#         target - 4x4 numpy array representing the desired transformation from
#         end effector to world
#         current - 4x4 numpy array representing the "current" end effector orientation

#         OUTPUTS:
#         displacement - a 3-element numpy array containing the displacement from
#         the current frame to the target frame, expressed in the world frame
#         axis - a 3-element numpy array containing the axis of the rotation from
#         the current frame to the end effector frame. The magnitude of this vector
#         must be sin(angle), where angle is the angle of rotation around this axis
#         """

#         ## STUDENT CODE STARTS HERE
#         displacement = target[:3,3] - current[:3,3]
#         R_cur = current[:3,:3]
#         R_tar = target[:3,:3]
#         axis = calcAngDiff(R_tar,R_cur)

#         ## END STUDENT CODE
#         return displacement, axis

#     @staticmethod
#     def distance_and_angle(G, H):
#         """
#         Helper function which computes the distance and angle between any two
#         transforms.

#         This data can be used to decide whether two transforms can be
#         considered equal within a certain linear and angular tolerance.

#         Be careful! Using the axis output of displacement_and_axis to compute
#         the angle will result in incorrect results when |angle| > pi/2

#         INPUTS:
#         G - a 4x4 numpy array representing some homogenous transformation
#         H - a 4x4 numpy array representing some homogenous transformation

#         OUTPUTS:
#         distance - the distance in meters between the origins of G & H
#         angle - the angle in radians between the orientations of G & H
#         """
        
#         ## STUDENT CODE STARTS HERE
#         # Extract translation vectors
#         p_G = G[:3, 3]
#         p_H = H[:3, 3]
        
#         # Calculate distance
#         distance = np.linalg.norm(p_G - p_H)
        
#         # Extract rotation matrices
#         R_G = G[:3, :3]
#         R_H = H[:3, :3]
        
#         # Calculate relative rotation matrix
#         R = np.dot(R_H.T, R_G)
        
#         # Calculate angle using trace formula, ensuring the value is in [-1, 1]
#         trace_R = np.trace(R)
#         cos_theta = (trace_R - 1) / 2
#         cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Constrain to avoid numerical errors
        
#         angle = np.arccos(cos_theta)

#         ## END STUDENT CODE
#         return distance, angle

#     def is_valid_solution(self,q,target):
#         """
#         Given a candidate solution, determine if it achieves the primary task
#         and also respects the joint limits.

#         INPUTS
#         q - the candidate solution, namely the joint angles
#         target - 4x4 numpy array representing the desired transformation from
#         end effector to world

#         OUTPUTS:
#         success - a Boolean which is True if and only if the candidate solution
#         produces an end effector pose which is within the given linear and
#         angular tolerances of the target pose, and also respects the joint
#         limits.
#         """

#         ## STUDENT CODE STARTS HERE
#         success = False
#         message = " "
#         # Check joint limits
#         if np.any(q < self.lower) or np.any(q > self.upper):
#             message = "Solution not found, Joint angles out of bounds"
#             return success, message
#         # Compute the achieved end-effector transformation using forward kinematics
#         _,achieved = self.fk.forward(q)
        
#         # Calculate the distance and angle between achieved and target transformations
#         distance, angle = self.distance_and_angle(achieved, target)
        
#         # Check if within linear and angular tolerances
#         if distance > self.linear_tol:
#             message = "Solution not found, Position tolerance exceeded"
#             return success, message

#         if angle > self.angular_tol:
#             message = "Solution not found, Orientation tolerance exceeded"
#             return success, message
        
#         success = True
#         message = "Solution found"
#         ## END STUDENT CODE
#         return success, message

#     ####################
#     ## Task Functions ##
#     ####################

#     @staticmethod
#     def end_effector_task(q,target,method):
#         """
#         Primary task for IK solver. Computes a joint velocity which will reduce
#         the error between the target end effector pose and the current end
#         effector pose (corresponding to configuration q).

#         INPUTS:
#         q - the current joint configuration, a "best guess" so far for the final answer
#         target - a 4x4 numpy array containing the desired end effector pose
#         method - a boolean variable that determines to use either 'J_pseudo' or 'J_trans' 
#         (J pseudo-inverse or J transpose) in your algorithm
        
#         OUTPUTS:
#         dq - a desired joint velocity to perform this task, which will smoothly
#         decay to zero magnitude as the task is achieved
#         """

#         ## STUDENT CODE STARTS HERE
#         dq = np.zeros(7)
        
#         _,current = IK.fk.forward(q)
#         displacement, axis = IK.displacement_and_axis(target, current)
        
#         # Concatenate displacement and axis to form the desired end-effector velocity
#         v_desired = np.hstack((displacement, axis))
        
#         # Step 2: Compute the Jacobian matrix at the current configuration
#         J = calcJacobian(q)
        
#         # Step 3: Compute dq based on the specified method
#         if method == 'J_pseudo':  # If method is True, use the pseudo-inverse
#             dq = np.dot(np.linalg.pinv(J), v_desired)
#         if method == 'J_trans':  # If method is False, use the Jacobian transpose
#             dq = np.dot(J.T, v_desired)

#         ## END STUDENT CODE
#         return dq

#     @staticmethod
#     def joint_centering_task(q,rate=5e-1): 
#         """
#         Secondary task for IK solver. Computes a joint velocity which will
#         reduce the offset between each joint's angle and the center of its range
#         of motion. This secondary task acts as a "soft constraint" which
#         encourages the solver to choose solutions within the allowed range of
#         motion for the joints.

#         INPUTS:
#         q - the joint angles
#         rate - a tunable parameter dictating how quickly to try to center the
#         joints. Turning this parameter improves convergence behavior for the
#         primary task, but also requires more solver iterations.

#         OUTPUTS:
#         dq - a desired joint velocity to perform this task, which will smoothly
#         decay to zero magnitude as the task is achieved
#         """

#         # normalize the offsets of all joints to range from -1 to 1 within the allowed range
#         offset = 2 * (q - IK.center) / (IK.upper - IK.lower)
#         dq = rate * -offset # proportional term (implied quadratic cost)

#         return dq
        
#     ###############################
#     ## Inverse Kinematics Solver ##
#     ###############################

#     def inverse(self, target, seed, method, alpha):
#         """
#         Uses gradient descent to solve the full inverse kinematics of the Panda robot.

#         INPUTS:
#         target - 4x4 numpy array representing the desired transformation from
#         end effector to world
#         seed - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6], which
#         is the "initial guess" from which to proceed with optimization
#         method - a boolean variable that determines to use either 'J_pseudo' or 'J_trans' 
#         (J pseudo-inverse or J transpose) in your algorithm

#         OUTPUTS:
#         q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6], giving the
#         solution if success is True or the closest guess if success is False.
#         success - True if the IK algorithm successfully found a configuration
#         which achieves the target within the given tolerance. Otherwise False
#         rollout - a list containing the guess for q at each iteration of the algorithm
#         """

#         q = seed
#         rollout = []

#         ## STUDENT CODE STARTS HERE
        
#         ## gradient descent:
#         while True:
#             rollout.append(q)

#             # Primary Task - Achieve End Effector Pose
#             dq_ik = IK.end_effector_task(q,target, method)

#             # Secondary Task - Center Joints
#             dq_center = IK.joint_centering_task(q)

#             ## Task Prioritization
#             J = calcJacobian(q)

#             if method == 'J_pseudo':
#                 J_inv = np.linalg.pinv(J)
#             if method == 'J_trans' :
#                 J_inv = J.T@np.linalg.inv(J@J.T)
#             # Calculate dq using null space projection
#             null_space_projector = np.eye(7) - J_inv @ J
#             dq = dq_ik + null_space_projector @ dq_center
#             # Check termination conditions
#             if np.linalg.norm(dq) < self.min_step_size:
#                 break
#             if len(rollout) > self.max_steps:
#                 break
#             # update q
#             q = q + alpha *dq
#         ## END STUDENT CODE

#         success, message = self.is_valid_solution(q,target)
#         return q, rollout, success, message

# ################################
# ## Simple Testing Environment ##
# ################################

# if __name__ == "__main__":

#     np.set_printoptions(suppress=True,precision=5)

#     ik = IK()

#     # matches figure in the handout
#     seed = np.array([0,0,0,-pi/2,0,pi/2,pi/4])

#     target_red = np.array([
#         [1,0,0,0.532],
#         [0,-1,0,-0.81],
#         [0,0,-1,0.6],
#         [0,0,0, 1],
#     ])
#     target = np.array([
#         [0,-1,0,0.2],
#         [-1,0,0,0],
#         [0,0,-1,.3],
#         [0,0,0, 1],
#     ])
#     T_b_w = np.array([[1,0,0,0],\
#                         [0,1,0,0.990],\
#                         [0,0,1,0],\
#                         [0,0,0,1]])
#     target_r = T_b_w @ target_red
#     # Using pseudo-inverse 
#     q_pseudo, rollout_pseudo, success_pseudo, message_pseudo = ik.inverse(target_r, seed, method='J_pseudo', alpha=.5)

#     for i, q_pseudo in enumerate(rollout_pseudo):
#         joints, pose = ik.fk.forward(q_pseudo)
#         d, ang = IK.distance_and_angle(target,pose)
#         print('iteration:',i,' q =',q_pseudo, ' d={d:3.4f}  ang={ang:3.3f}'.format(d=d,ang=ang))

#     # Using pseudo-inverse 
#     # q_trans, rollout_trans, success_trans, message_trans = ik.inverse(target, seed, method='J_trans', alpha=.5)

#     # for i, q_trans in enumerate(rollout_trans):
#     #     joints, pose = ik.fk.forward(q_trans)
#     #     d, ang = IK.distance_and_angle(target,pose)
#     #     print('iteration:',i,' q =',q_trans, ' d={d:3.4f}  ang={ang:3.3f}'.format(d=d,ang=ang))

#     # compare
#     print("\n method: J_pseudo-inverse")
#     print("   Success: ",success_pseudo, ":  ", message_pseudo)
#     print("   Solution: ",q_pseudo)
#     print("   #Iterations : ", len(rollout_pseudo))
#     # print("\n method: J_transpose")
#     # print("   Success: ",success_trans, ":  ", message_trans)
#     # print("   Solution: ",q_trans)
#     # print("   #Iterations :", len(rollout_trans),'\n')
