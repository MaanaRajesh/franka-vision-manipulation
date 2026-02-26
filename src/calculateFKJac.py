import numpy as np
from math import pi
import copy

class FK_Jac():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab 1 and 4 handout

        self.DH_params = [
            # [a, alpha, d, theta]
            [0, 0, 0.141, np.pi],    # Joint 1
            [0, np.pi/2, 0.192, 0], # Joint 2
            [0, -np.pi/2, 0, 0],      # Joint 3
            [-0.0825, -np.pi/2, 0.316, 0],   # Joint 4
            [-0.0825, -np.pi/2, 0, np.pi],   # Joint 5
            [0, 0, 0.384, 0],
            [0, np.pi/2, 0, 0],     # Joint 6
            [0.088, np.pi/2, 0, 0],  # Joint 7
            [0, 0, 0.210, -np.pi/4]         # Joint e (end effector)
        ]

        self.offset = [
            np.array([0, 0, 0]),     # Offset in later portion
            np.array([0, 0, 0]),
            np.array([0, 0, 0.195]),
            np.array([0, 0, 0]),
            np.array([0, 0, 0.125]),
            np.array([0, 0, 0]),
            np.array([0, 0, -0.015]),
            np.array([0, 0, 0.051]),
            np.array([0, 0, 0])
        ]


    def dh_transform(self, theta, d, a, alpha):
        """ Compute individual transformation matrix using DH parameters. """
        T = np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha),  np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
            [np.sin(theta),  np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
            [0,              np.sin(alpha),                 np.cos(alpha),                 d],
            [0,              0,                             0,                             1]
        ])
        return T

    def forward_expanded(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -10 x 3 matrix, where each row corresponds to a physical or virtual joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 10 x 4 x 4 homogeneous transformation matrix,
                  representing the each joint/end effector frame expressed in the
                  world frame
        """

        # Your code starts here

        jointPositions = np.zeros((10,3))
        T0e = np.zeros((10,4,4))
        
        # Start with the base frame at the origin
        T = np.identity(4)
        T0e[0, :, :] = T

        # Loop through each joint to compute positions and transformations
        for i in range(len(self.DH_params)):
            if i == 0:
                a = self.DH_params[i][0]
                alpha = self.DH_params[i][1]
                d = self.DH_params[i][2]
                theta = self.DH_params[i][3]  # Add joint angle to DH theta
                
                # Compute the transformation from the previous joint to the current joint
                T_joint = self.dh_transform(theta, d, a, alpha)
                
                # Update the overall transformation matrix T0e
                T = np.dot(T, T_joint)

            elif i == 5:
                a = self.DH_params[i][0]
                alpha = self.DH_params[i][1]
                d = self.DH_params[i][2]
                theta = self.DH_params[i][3]  # Add joint angle to DH theta
                
                # Compute the transformation from the previous joint to the current joint
                T_joint = self.dh_transform(theta, d, a, alpha)
                
                # Update the overall transformation matrix T0e
                T = np.dot(T, T_joint)
            elif i < 5:
                a = self.DH_params[i][0]
                alpha = self.DH_params[i][1]
                d = self.DH_params[i][2]
                theta = q[i-1] + self.DH_params[i][3]  # Add joint angle to DH theta
                
                # Compute the transformation from the previous joint to the current joint
                T_joint = self.dh_transform(theta, d, a, alpha)
                
                # Update the overall transformation matrix T0e
                T = np.dot(T, T_joint)
                
                # Extract the position of the current joint and save it
                jointPositions[i, :] = T[0:3, 3]  # Take the translation part
                if self.offset[i].any():
                    jointPositions[i, :] += np.dot(T[0:3,0:3], self.offset[i])
                    T1 = copy.deepcopy(T)
                    T1[0:3, 3] += np.dot(T1[0:3,0:3], self.offset[i])
                    T0e[i, :, :] = T1
                else:
                    T0e[i, :, :] = T
            else:
                a = self.DH_params[i][0]
                alpha = self.DH_params[i][1]
                d = self.DH_params[i][2]
                theta = q[i-2] + self.DH_params[i][3]  # Add joint angle to DH theta
                
                # Compute the transformation from the previous joint to the current joint
                T_joint = self.dh_transform(theta, d, a, alpha)
                
                # Update the overall transformation matrix T0e
                T = np.dot(T, T_joint)
                
                # Extract the position of the current joint and save it
                jointPositions[i-1, :] = T[0:3, 3]  # Take the translation part
                if self.offset[i].any():
                    jointPositions[i-1, :] += np.dot(T[0:3,0:3], self.offset[i])
                    T1 = copy.deepcopy(T)
                    T1[0:3, 3] += np.dot(T1[0:3,0:3], self.offset[i])
                    T0e[i-1, :, :] = T1
                else:
                    T0e[i-1, :, :] = T

        jointPositions[0, 2] += 0.141
        T0e[0, 2, 3] += 0.141

        jointPositions[-2, :] = jointPositions[-3, :] + np.dot(T[0:3, 0:3], [0,0.1,-0.105])
        jointPositions[-1, :] = jointPositions[-3, :] + np.dot(T[0:3, 0:3], [0,-0.1,-0.105])
        T0e[-2, 0:3, 3] = jointPositions[-2, :]
        T0e[-1, 0:3, 3] = jointPositions[-1, :]
        T0e[-2, 0:3, 0:3] = T[0:3, 0:3]
        T0e[-1, 0:3, 0:3] = T[0:3, 0:3]
        
        # Your code ends here

        return jointPositions, T0e

    # feel free to define additional helper methods to modularize your solution for lab 1

    
    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        # STUDENT CODE HERE

        return()
    
    def calcJacobian(self, q_in, joint_num):
        """
        Calculate the full Jacobian of the end effector in a given configuration
        :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
        :return: J - 6 x 9 matrix representing the Jacobian, where the first three
        rows correspond to the linear velocity and the last three rows correspond to
        the angular velocity, expressed in world frame coordinates
        """

        J = np.zeros((6, 9))

        ## STUDENT CODE GOES HERE
        
        # Perform forward kinematics to get joint positions and the end-effector transformation matrix
        joint_positions, T0e = self.forward_expanded(q_in)  # Joint positions is a 8x3 matrix

        # Extract the position of the last joint
        p_i = T0e[joint_num, :3, 3]

        # Initialize transformation matrix from the base frame
        T = np.identity(4)  # Start from the base frame
        z = np.array([0, 0, 1])  # Initial z-axis direction (aligned with the base frame)

        # Compute Jacobian columns for each joint
        j = 0
        loop_size = joint_num if joint_num <= 7 else 7
        for k in range(loop_size):
            # Update the transformation matrix for the next joint
            a = self.DH_params[j][0]
            alpha = self.DH_params[j][1]
            d = self.DH_params[j][2]
            if j == 0 or j == 5:
                theta = self.DH_params[j][3]  # Add joint angle to DH theta
            else:
                theta = q_in[k-1] + self.DH_params[j][3]  # Add joint angle to DH theta
            T_joint = self.dh_transform(theta, d, a, alpha)
            T = np.dot(T, T_joint)

            if j == 5:
                j = j + 1
                a = self.DH_params[j][0]
                alpha = self.DH_params[j][1]
                d = self.DH_params[j][2]
                theta = q_in[k-1] + self.DH_params[j][3]  # Add joint angle to DH theta
                T_joint = self.dh_transform(theta, d, a, alpha)
                T = np.dot(T, T_joint)

            # Extract the current joint position
            p_k = joint_positions[k, :]  # Position of the ith joint
            
            # Extract the rotation axis (z-axis of the current joint's frame)
            z_k = T[0:3, 2]  # z-axis of the current joint (rotation axis)

            # Linear velocity component
            J_v = np.cross(z_k, (p_i - p_k))

            # Angular velocity component for revolute joints
            J_w = z_k

            # Set the Jacobian columns (first 3 rows for linear, last 3 rows for angular velocity)
            J[0:3, k] = J_v  # Linear velocity part
            J[3:6, k] = J_w  # Angular velocity part

            j = j + 1
        
        if joint_num == 8:
            p_k = joint_positions[7, :]
            J_v = np.cross(z_k, (p_i - p_k))
            J_w = z_k
            J[0:3, joint_num-1] = J_v  # Linear velocity part
            J[3:6, joint_num-1] = J_w  # Angular velocity part
        if joint_num == 9:
            p_k = joint_positions[7, :]
            J_v = np.cross(z_k, (p_i - p_k))
            J_w = z_k
            J[0:3, joint_num-2] = J_v  # Linear velocity part
            J[3:6, joint_num-2] = J_w  # Angular velocity part
            p_k = joint_positions[8, :]
            J_v = np.cross(z_k, (p_i - p_k))
            J_w = z_k
            J[0:3, joint_num-1] = J_v  # Linear velocity part
            J[3:6, joint_num-1] = J_w  # Angular velocity part
        return J
    
if __name__ == "__main__":

    fk = FK_Jac()

    # matches figure in the handout
    q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])

    joint_positions, T0e = fk.forward_expanded(q)

    for i in range(10):
        Jv = fk.calcJacobian(q, i)[0:3]
        print(f"i: {i}, J_v: \n {np.round(Jv,3)}")
    
    # print("Joint Positions:\n",joint_positions)
    # print("End Effector Pose:\n",T0e)
