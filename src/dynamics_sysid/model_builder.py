import pinocchio as pin
import sys

# Add src and config directories to python path
sys.path.append('src')
sys.path.append('config')
sys.path.append('utils')
from utils import MathUtils
from config import robot_config
import numpy as np

def build_model_from_kdl():
    """
    Builds a Pinocchio model programmatically from the KDL chain and dynamic
    parameters specified in the robot_config file. This ensures the model
    perfectly matches the trusted "Neocis convention" kinematics.
    
    This model will be a fixed-base manipulator.
    """
    model = pin.Model()
    model.name = robot_config.ROBOT_NAME # Set the robot's name

    kdl_params = np.reshape(np.asarray(robot_config.KDL_CHAIN, dtype="float"), (-1, 6))
    
    # --- Step 1: Define the Base Link ---
    # The first body is attached to the "universe" joint (joint 0).
    # Its placement is relative to the world origin.
    base_placement = MathUtils.convert_to_homogeneous_transform(kdl_params[0], neocis_convention=True)
    base_inertia_dict = robot_config.LINK_DYNAMIC_PARAMETERS["Base"]
    
    I_base = pin.Inertia(
        base_inertia_dict['mass'],
        base_inertia_dict['com'],
        np.array(base_inertia_dict['inertia_tensor'])
    )
    
    # Add the base body to the universe (joint 0).
    # Note: Pinocchio v3 automatically names the first joint "universe".
    # The first body has no explicit name in the model structure but is associated with this joint.
    model.appendBodyToJoint(0, I_base, pin.SE3(base_placement))

    # --- Step 2: Iteratively add links and revolute joints ---
    parent_joint_id = 0 # Start with the universe joint

    for i in range(robot_config.NUM_JOINTS):
        # Retrieve names and parameters for the current joint and link
        joint_name = robot_config.ACTUATED_JOINT_NAMES[i]
        link_name = robot_config.LINK_NAMES_IN_KDL_ORDER[i + 1] # +1 because index 0 is the Base
        
        # Placement of the new joint frame relative to the parent joint frame
        joint_placement = MathUtils.convert_to_homogeneous_transform(kdl_params[i + 1], neocis_convention=True)
        M = pin.SE3(joint_placement)
        
        # Inertia of the new link, defined in its own link frame
        link_dynamics_dict = robot_config.LINK_DYNAMIC_PARAMETERS[link_name]
        inertia = pin.Inertia(
            link_dynamics_dict['mass'],
            link_dynamics_dict['com'],
            np.array(link_dynamics_dict['inertia_tensor'])
        )
        
        # Add the joint. This also automatically adds the joint name to 'model.names'.
        joint_id = model.addJoint(parent_joint_id, pin.JointModelRZ(), M, joint_name)
        
        # Append the body (link) to the newly created joint. The body's inertia is attached.
        model.appendBodyToJoint(joint_id, inertia, pin.SE3.Identity())
        
        # Update the parent for the next iteration
        parent_joint_id = joint_id

    # --- Step 3: Handle the fixed End_Effector to Weight/Sensor Flange joint ---
    # The final link in the chain is a composite of the End_Effector and the Weight.
    T_ee_weight = MathUtils.convert_to_homogeneous_transform(kdl_params[-1], neocis_convention=True)
    
    ee_dynamics_dict = robot_config.LINK_DYNAMIC_PARAMETERS["End_Effector"]
    weight_dynamics_dict = robot_config.LINK_DYNAMIC_PARAMETERS["Weight"]
    
    I_ee = pin.Inertia(
        ee_dynamics_dict['mass'], 
        ee_dynamics_dict['com'], 
        np.array(ee_dynamics_dict['inertia_tensor'])
    )
    I_weight = pin.Inertia(
        weight_dynamics_dict['mass'], 
        weight_dynamics_dict['com'], 
        np.array(weight_dynamics_dict['inertia_tensor'])
    )
    
    # Combine the end effector and weight inertias.
    # The inertia of the weight must be transformed into the end-effector's frame before being added.
    composite_inertia = I_ee + pin.SE3(T_ee_weight).act(I_weight)
    
    # The last body's inertia was already added in the loop. We now UPDATE it to be the composite inertia.
    # `model.inertias` includes the inertia of the universe, so the last actual link is at index -1.
    model.inertias[-1] = composite_inertia

    # --- Step 4: Finalize and Review Model ---
    data = model.createData()

    print("--- Pinocchio Model Built from KDL ---")
    print(f"Model Name: {model.name}")
    print(f"Number of joints (including universe): {model.njoints}")
    print(f"Number of bodies (including universe): {model.nbodies}")
    print(f"Number of generalized positions (q): {model.nq}")
    print(f"Number of generalized velocities (v): {model.nv}")
    # model.names contains the names of the joints.
    print(f"Joint names: {model.names}")
    print("--------------------------------------")
    
    return model, data