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