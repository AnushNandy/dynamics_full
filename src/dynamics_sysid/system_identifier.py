import numpy as np
import pinocchio as pin

def identify_dynamic_parameters(model, data, q_data, v_data, a_data, tau_data):
    """
    Identifies the 10 inertial parameters per link using least squares.
    """
    num_samples = len(q_data)
    num_links = model.nv
    num_params_per_link = 10
    num_joints = model.nv

    # The regressor matrix W will be (num_samples * num_joints) x (num_links * 10)
    # This shape will now be (num_samples * 7) x (7 * 10) = (num_samples * 7) x 70
    W = np.zeros((num_samples * num_joints, num_links * num_params_per_link))
    T = tau_data.flatten()
    
    print("\n--- Building Regressor Matrix ---")
    for i in range(num_samples):
        # This regressor correctly has a shape of (7, 70)
        regressor = pin.computeJointTorqueRegressor(model, data, q_data[i], v_data[i], a_data[i])
        start_row = i * num_joints
        end_row = start_row + num_joints
        W[start_row:end_row, :] = regressor
    
    print("Regressor matrix built. Shape:", W.shape)
    print("Solving the least squares problem W * Phi = T...")
    # Use a small rcond to prevent warnings and handle potential ill-conditioning
    phi, residuals, rank, s = np.linalg.lstsq(W, T, rcond=None)

    print("Least squares solved.")
    return phi.flatten()

def validate_model(model, data, identified_params, q_val, v_val, a_val, tau_val):
    """
    Validates the identified model on a new dataset.
    """
    num_samples = len(q_val)
    tau_predicted = np.zeros_like(tau_val)

    print("\n--- Validating Model ---")
    for i in range(num_samples):
        regressor = pin.computeJointTorqueRegressor(model, data, q_val[i], v_val[i], a_val[i])
        tau_predicted[i, :] = regressor @ identified_params
        
    error = tau_val - tau_predicted
    rmse = np.sqrt(np.mean(error**2))
    
    return rmse, tau_predicted

def update_model_with_identified_params(model, identified_params_flat):
    """
    Updates the inertias of a Pinocchio model with a flat vector of identified parameters.
    
    Args:
        model (pin.Model): The Pinocchio model to update.
        identified_params_flat (np.ndarray): A flat array of size (num_links * 10).
    
    Returns:
        pin.Model: The updated model.
    """
    num_links = model.nv
    if len(identified_params_flat) != num_links * 10:
        raise ValueError("Size of identified_params_flat is incorrect.")

    # Note: model.inertias[0] is for the universe, so we start from index 1.
    for i in range(num_links):
        link_params = identified_params_flat[i*10 : (i+1)*10]
        # Create a new Inertia object from the 10 dynamic parameters
        # [m, mc_x, mc_y, mc_z, I_xx, I_xy, I_yy, I_xz, I_yz, I_zz]
        inertia = pin.Inertia.FromDynamicParameters(link_params)
        model.inertias[i + 1] = inertia
    
    print("\nPinocchio model updated with identified inertial parameters.")
    return model


def compute_pd_plus_gravity_control(model, data, q_des, v_des, q_act, v_act, Kp, Kd):
    """
    Computes torque using PD control plus feedforward gravity compensation.
    
    Args:
        model (pin.Model): The Pinocchio model (with identified parameters).
        data (pin.Data): The Pinocchio data object.
        q_des, v_des (np.ndarray): Desired joint position and velocity.
        q_act, v_act (np.ndarray): Actual joint position and velocity.
        Kp, Kd (np.ndarray): Proportional and derivative gains (diagonal matrices).
        
    Returns:
        np.ndarray: The computed joint torques.
    """
    # Pinocchio's RNEA requires q, v to compute gravity term correctly
    pin.forwardKinematics(model, data, q_act, v_act)

    # 1. Gravity Compensation Term (Feedforward)
    tau_gravity = pin.computeGeneralizedGravity(model, data, q_act)

    # 2. PD Control Term (Feedback)
    pos_error = q_des - q_act
    vel_error = v_des - v_act
    tau_pd = Kp @ pos_error + Kd @ vel_error

    # 3. Total Torque
    tau_command = tau_pd + tau_gravity
    
    return tau_command