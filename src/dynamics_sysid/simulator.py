import pybullet as p
import pybullet_data
import time
import numpy as np
import sys

# Add src and config directories to python path
sys.path.append('src')
sys.path.append('config')
from config import robot_config

class PhysicsSimulator:
    """
    A wrapper class for the PyBullet physics simulation environment.
    """
    def __init__(self, urdf_path):
        """
        Initializes the simulator, loads the robot, and sets up the environment.

        Args:
            urdf_path (str): Path to the robot's URDF file.
        """
        # Connect to the physics server
        self.client_id = p.connect(p.GUI) # Use p.DIRECT for non-GUI mode
        
        # Configure the GUI
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load environment assets
        p.loadURDF("plane.urdf")
        
        # Load the robot
        start_pos = [0, 0, 1]  # x, y, z position (z = height)
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF(urdf_path, basePosition=start_pos, useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)
        
        # Build a mapping from joint names to PyBullet's joint indices
        self.joint_name_to_id = {}
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('UTF-8')
            if joint_name in robot_config.ACTUATED_JOINT_NAMES:
                self.joint_name_to_id[joint_name] = i
        
        # Get the list of actuated joint indices in the correct order
        self.actuated_joint_ids = [self.joint_name_to_id[name] for name in robot_config.ACTUATED_JOINT_NAMES]
        
        # REMOVED: The conflicting velocity control setup that was locking joints

        print("--- PyBullet Simulator Initialized ---")
        print(f"Robot ID: {self.robot_id}")
        print(f"Actuated Joint Names to IDs: {self.joint_name_to_id}")
        print("------------------------------------")

    def set_joint_positions(self, q, wait_for_convergence=False):
        """
        Sets the robot's joint positions for visualization or control.
        
        Args:
            q: Target joint positions
            wait_for_convergence: If True, waits until joints reach target positions
        """
        # Use position control with reasonable gains
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.actuated_joint_ids,
            controlMode=p.POSITION_CONTROL,
            targetPositions=q,
            forces=[500] * len(self.actuated_joint_ids),  # Max force
            positionGains=[0.1] * len(self.actuated_joint_ids),  # P gain
            velocityGains=[0.1] * len(self.actuated_joint_ids)   # D gain
        )
        
        if wait_for_convergence:
            # Wait until robot reaches approximately the target position
            max_iterations = 100
            tolerance = 0.01  # radians
            
            for _ in range(max_iterations):
                p.stepSimulation()
                current_positions = self.get_joint_positions()
                if np.allclose(current_positions, q, atol=tolerance):
                    break
                time.sleep(1./240.)

    def get_joint_positions(self):
        """Returns current joint positions."""
        joint_states = p.getJointStates(self.robot_id, self.actuated_joint_ids)
        return np.array([state[0] for state in joint_states])

    def step_simulation(self):
        """Advances the simulation by one time step."""
        p.stepSimulation()

    def compute_inverse_dynamics(self, q, v, a):
        """
        Computes the joint torques required to achieve the given state (q, v, a).
        This serves as our "ground truth" torque measurement from the simulator.
        """
        torques = p.calculateInverseDynamics(
            self.robot_id,
            list(q),
            list(v),
            list(a)
        )
        actuated_torques = [torques[i] for i in self.actuated_joint_ids]
        return np.array(actuated_torques)

    def disconnect(self):
        """Disconnects from the physics server."""
        p.disconnect()