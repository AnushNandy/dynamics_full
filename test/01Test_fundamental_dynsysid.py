import numpy as np
import time
import sys
import pybullet as p
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)
from src.dynamics_sysid import model_builder, trajectory_generator, data_processing, system_identifier
from utils import plotting_utils
from src.dynamics_sysid.simulator import PhysicsSimulator
from config import robot_config

def run_interactive_mode(simulator):
    """Allows user to control the robot with GUI sliders."""
    print("\n--- Interactive Control Mode ---")
    print("Use the sliders in the PyBullet window to control the robot.")
    print("Close the PyBullet window or press Ctrl+C in the terminal to exit.")
    
    # Create debug sliders for each joint
    joint_sliders = []
    for i, name in enumerate(robot_config.ACTUATED_JOINT_NAMES):
        slider = p.addUserDebugParameter(
            name,
            rangeMin=-np.pi,
            rangeMax=np.pi,
            startValue=0
        )
        joint_sliders.append(slider)
        
    try:
        while True:
            # Read slider values
            target_positions = [p.readUserDebugParameter(slider) for slider in joint_sliders]
            
            # Apply positions using the simulator's built-in PD controller
            simulator.set_joint_positions(target_positions)
            
            # Step simulation
            simulator.step_simulation()
            time.sleep(1./240.)
            
    except p.error:
        print("PyBullet window closed. Exiting interactive mode.")
    except KeyboardInterrupt:
        print("\nExiting interactive mode.")


def main():
    """Main pipeline for system identification and control validation."""
    
    # --- Setup ---
    print("Building Pinocchio model from KDL config...")
    model, data = model_builder.build_model_from_kdl()

    print("Initializing PyBullet simulator...")
    urdf_path = r"/home/robot/dev/dynamics_full/ArmModels/urdfs/P4/P4_Contra-Angle_right.urdf"
    simulator = PhysicsSimulator(urdf_path)
    
    # --- User Mode Selection ---
    print("\n--- Mode Selection ---")
    print("1: Run interactive control mode.")
    print("2: Run full System ID and Control Validation pipeline.")
    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        run_interactive_mode(simulator)
        simulator.disconnect()
        return
    elif choice != '2':
        print("Invalid choice. Exiting.")
        simulator.disconnect()
        return

    # --- PART 1: SYSTEM IDENTIFICATION ---
    print("\n--- PART 1: System Identification ---")

    # 1. Generate Training Trajectory
    print("\nGenerating training trajectory...")
    t_train, q_train, v_train, a_train = trajectory_generator.generate_excitation_trajectory(
        model, num_points=5000, duration=20
    )

    # 2. Collect Data
    q_meas_train, v_proc_train, a_proc_train, tau_meas_train = data_processing.collect_data_from_simulator(
        simulator, q_train, v_train, a_train
    )

    # 3. Identify Parameters
    identified_parameters = system_identifier.identify_dynamic_parameters(
        model, data, q_meas_train, v_proc_train, a_proc_train, tau_meas_train
    )
    
    # 4. Validation
    print("\n--- Identification Validation ---")
    t_val, q_val, v_val, a_val = trajectory_generator.generate_excitation_trajectory(
        model, num_points=2000, duration=8
    )
    q_meas_val, v_proc_val, a_proc_val, tau_meas_val = data_processing.collect_data_from_simulator(
        simulator, q_val, v_val, a_val
    )
    validation_rmse, tau_predicted_val = system_identifier.validate_model(
        model, data, identified_parameters, q_meas_val, v_proc_val, a_proc_val, tau_meas_val
    )
    print(f"\nValidation RMSE of Identified Model: {validation_rmse:.6f} Nm")
    
    # --- Plotting SysID Results ---
    plotting_utils.plot_sysid_validation(t_val, tau_meas_val, tau_predicted_val, robot_config.ACTUATED_JOINT_NAMES)
    
    initial_params_from_model = np.concatenate([inertia.toDynamicParameters() for inertia in model.inertias[1:]])
    link_names_for_plot = robot_config.LINK_NAMES_IN_KDL_ORDER[1:8] # Links 0-6
    plotting_utils.plot_parameter_comparison(link_names_for_plot, initial_params_from_model, identified_parameters)
    
    input("\nSystem ID complete. Press Enter to proceed to Control Validation...")

    # --- PART 2: CONTROL WITH IDENTIFIED MODEL ---
    print("\n--- PART 2: Control with Identified Model ---")
    # 1. Update Pinocchio model with the parameters we just found
    model_identified = system_identifier.update_model_with_identified_params(model, identified_parameters)
    data_identified = model_identified.createData()

    # 2. Set up controller
    num_joints = model.nv
    Kp = np.diag([100.0] * num_joints)  # Proportional gains
    Kd = np.diag([10.0] * num_joints)   # Derivative gains
    print(f"Controller gains set. Kp={Kp[0,0]}, Kd={Kd[0,0]}")

    # 3. Reset simulator to the start of the validation trajectory
    print("Resetting robot to trajectory start position...")
    simulator.set_joint_positions(q_val[0], wait_for_convergence=True)
    time.sleep(1.0) # Settle

    # 4. Run the control loop
    print("Running PD + Gravity Compensation control loop...")
    num_steps = len(t_val)
    q_actual_hist = np.zeros_like(q_val)
    v_actual_hist = np.zeros_like(v_val)
    tau_cmd_hist = np.zeros_like(q_val)

    for i in range(num_steps):
        # Get current state from simulator
        q_act, v_act = simulator.get_joint_states()

        # Get desired state from validation trajectory
        q_des, v_des = q_val[i], v_val[i]

        # Compute control torque using our identified model
        tau_cmd = system_identifier.compute_pd_plus_gravity_control(
            model_identified, data_identified, q_des, v_des, q_act, v_act, Kp, Kd
        )

        # Apply torques to the robot
        simulator.apply_joint_torques(tau_cmd)

        # Step simulation
        simulator.step_simulation()
        
        # Record data for plotting
        q_actual_hist[i, :] = q_act
        v_actual_hist[i, :] = v_act
        tau_cmd_hist[i, :] = tau_cmd
        
        time.sleep(1./240.) # Match simulation frequency if needed

    print("Control simulation finished.")

    # --- Plotting Control Results ---
    plotting_utils.plot_control_performance(t_val, q_val, q_actual_hist, v_val, v_actual_hist, robot_config.ACTUATED_JOINT_NAMES)
    plotting_utils.plot_control_torques(t_val, tau_cmd_hist, robot_config.ACTUATED_JOINT_NAMES)

    # --- Cleanup ---
    simulator.disconnect()
    print("\nFull pipeline complete.")

if __name__ == '__main__':
    main()