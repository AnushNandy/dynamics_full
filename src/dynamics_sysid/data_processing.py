import numpy as np

def collect_data_from_simulator(simulator, q_traj, v_traj, a_traj):
    """
    Collects "measured" data by querying the PyBullet simulator.

    Args:
        simulator (PhysicsSimulator): An instance of our simulator class.
        q_traj (np.ndarray): Trajectory of joint positions.
        v_traj (np.ndarray): Trajectory of joint velocities.
        a_traj (np.ndarray): Trajectory of joint accelerations.

    Returns:
        tuple[np.ndarray, ...]: (q, v, a, tau) data arrays.
    """
    print("\n--- Collecting Data from PyBullet Simulator ---")
    num_samples = len(q_traj)
    num_joints = q_traj.shape[1]
    
    # In this ideal case, the robot perfectly tracks the commanded trajectory
    q_measured = q_traj
    v_processed = v_traj
    a_processed = a_traj

    tau_measured = np.zeros((num_samples, num_joints))
    
    for i in range(num_samples):
        # The core change: use PyBullet's inverse dynamics as the "sensor"
        tau_measured[i, :] = simulator.compute_inverse_dynamics(
            q_traj[i], v_traj[i], a_traj[i]
        )

    print("Data collection complete.")
    return q_measured, v_processed, a_processed, tau_measured