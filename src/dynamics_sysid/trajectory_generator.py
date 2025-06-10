import numpy as np

def generate_excitation_trajectory(model, num_points, duration):
    """
    Generates an excitation trajectory for a fixed-base manipulator.
    """
    t = np.linspace(0, duration, num_points)
    num_joints = model.nv # For a fixed-base model, nv == nq == num_joints
    
    # Trajectory parameters (tune these for your robot)
    num_harmonics = 5
    base_freq = 0.1 * 2 * np.pi  # rad/s

    q = np.zeros((num_points, num_joints))
    v = np.zeros((num_points, num_joints))
    a = np.zeros((num_points, num_joints))

    for j in range(num_joints):
        for i in range(1, num_harmonics + 1):
            amplitude = np.pi / (2 * i * (j + 1)) 
            phase = np.random.uniform(0, 2 * np.pi)
            freq = i * base_freq

            q[:, j] += amplitude * np.sin(freq * t + phase)
            v[:, j] += amplitude * freq * np.cos(freq * t + phase)
            a[:, j] -= amplitude * freq**2 * np.sin(freq * t + phase)

    return t, q, v, a