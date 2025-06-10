import numpy as np
import time
import sys
import matplotlib.pyplot as plt


sys.path.append('src')
sys.path.append('config')

from src.dynamics_sysid import model_builder
from src.dynamics_sysid import trajectory_generator
from src.dynamics_sysid import data_processing
from src.dynamics_sysid import system_identifier
from src.dynamics_sysid.simulator import PhysicsSimulator # Import our new class
from config import robot_config

def plot_validation_results(t, tau_measured, tau_predicted, joint_names):
    """
    Plots the comparison between measured and predicted torques for each joint.
    """
    num_joints = tau_measured.shape[1]
    fig, axes = plt.subplots(num_joints, 1, figsize=(12, 2 * num_joints), sharex=True)
    fig.suptitle('Validation: Measured vs. Predicted Joint Torques', fontsize=16)

    for i in range(num_joints):
        axes[i].plot(t, tau_measured[:, i], 'b-', label='Measured (PyBullet)')
        axes[i].plot(t, tau_predicted[:, i], 'r--', label='Predicted (Identified Model)')
        axes[i].set_ylabel('Torque (Nm)')
        axes[i].set_title(f'Joint: {joint_names[i]}')
        axes[i].grid(True)
        axes[i].legend()

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    print("\nDisplaying torque validation plot. Close the plot window to exit.")
    plt.show()

def main():
    """Main pipeline for system identification using PyBullet."""
    
    # --- Setup ---
    # 1. Build the Pinocchio model from our trusted KDL chain for math/regressor calculations
    model, data = model_builder.build_model_from_kdl()

    # 2. Initialize the PyBullet simulator for visualization and data generation
    # The URDF provides the "ground truth" dynamics for the simulation.
    urdf_path = r"/home/robot/dev/dynamics_full/ArmModels/urdfs/P4/P4_Contra-Angle_right.urdf" # Make sure this file exists
    simulator = PhysicsSimulator(urdf_path)

    # --- Trajectory Generation and Visualization ---
    print("\nGenerating training trajectory...")
    t_train, q_train, v_train, a_train = trajectory_generator.generate_excitation_trajectory(
        model, num_points=5000, duration=20
    )
    
    # IMPROVED: Better trajectory visualization
    print("\nVisualizing trajectory in PyBullet...")
    print("Use mouse to rotate the view. The robot will move through the trajectory.")
    
    # Move to starting position first
    simulator.set_joint_positions(q_train[0], wait_for_convergence=True)
    time.sleep(1.0)  # Give time to see the starting position
    
    # Visualize trajectory - subsample for better visualization
    step_size = 20  # Show every 20th point for smoother visualization
    for i in range(0, len(q_train), step_size):
        simulator.set_joint_positions(q_train[i])
        
        # Step simulation multiple times to let the robot move toward target
        for _ in range(10):
            simulator.step_simulation()
            time.sleep(1./240.)
        
        # Optional: Print progress
        if i % (step_size * 10) == 0:
            print(f"Progress: {i/len(q_train)*100:.1f}%")
    
    # Alternative: Interactive step-by-step visualization
    # Uncomment this section if you want to step through manually
    """
    print("\nStep-by-step visualization (press Enter to advance):")
    for i in range(0, len(q_train), 100):  # Every 100th point
        simulator.set_joint_positions(q_train[i], wait_for_convergence=True)
        input(f"Step {i//100 + 1}: Press Enter to continue...")
    """
    
    input("\nVisualization complete. Press Enter to continue to System Identification...")

    # Rest of the main function continues as before...
    # --- System Identification ---
    # 3. Collect "measured" data from the PyBullet simulation
    q_meas_train, v_proc_train, a_proc_train, tau_meas_train = data_processing.collect_data_from_simulator(
        simulator, q_train, v_train, a_train
    )

    # 4. Identify dynamic parameters using the collected data
    identified_parameters = system_identifier.identify_dynamic_parameters(
        model, data, q_meas_train, v_proc_train, a_proc_train, tau_meas_train
    )
    
    # --- Validation ---
    print("\n--- Validation Phase ---")
    # 5. Generate a *different* trajectory for validation
    t_val, q_val, v_val, a_val = trajectory_generator.generate_excitation_trajectory(
        model, num_points=2000, duration=8
    )

    # 6. Collect validation data from the simulator
    q_meas_val, v_proc_val, a_proc_val, tau_meas_val = data_processing.collect_data_from_simulator(
        simulator, q_val, v_val, a_val
    )

    # 7. Validate the newly identified model against the simulated ground truth
    validation_rmse, tau_predicted_val = system_identifier.validate_model(
        model, data, identified_parameters, q_meas_val, v_proc_val, a_proc_val, tau_meas_val
    )
    print(f"\nValidation RMSE of Identified Model: {validation_rmse:.6f} Nm")
    
    # 8. For comparison, get the RMSE of the original parameters from our KDL-built model
    original_params_from_model = np.concatenate([inertia.toDynamicParameters() for inertia in model.inertias[1:]])
    original_model_rmse, _ = system_identifier.validate_model(
        model, data, original_params_from_model, q_meas_val, v_proc_val, a_proc_val, tau_meas_val
    )
    print(f"Validation RMSE of Original Config Model: {original_model_rmse:.6f} Nm")
    
    if validation_rmse < original_model_rmse and validation_rmse < 1e-3:
        print("\nSUCCESS: The pipeline successfully recovered the dynamic parameters from the simulation.")
    else:
        print("\nWARNING: High RMSE suggests a potential issue in the regressor calculation or solver.")

    # --- FIX: Add Plotting ---
    plot_validation_results(t_val, tau_meas_val, tau_predicted_val, robot_config.ACTUATED_JOINT_NAMES)

    # --- Cleanup ---
    simulator.disconnect()
    print("\nSystem Identification pipeline complete.")

if __name__ == '__main__':
    main()