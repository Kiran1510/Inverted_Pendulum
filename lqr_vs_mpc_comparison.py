"""
Inverted Pendulum Control: LQR vs MPC Comparison
Demonstrates the differences between linear and nonlinear control approaches
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import solve_discrete_are
import time

# ============================================================================
# SYSTEM PARAMETERS
# ============================================================================

class PendulumParams:
    """Physical parameters of the cart-pole system"""
    def __init__(self):
        self.M = 1.0        # Cart mass [kg]
        self.m = 0.3        # Pendulum mass [kg]
        self.l = 0.5        # Pendulum length [m]
        self.g = 9.81       # Gravity [m/s^2]
        self.b = 0.1        # Friction coefficient
        self.dt = 0.02      # Time step [s]
        
        # Constraints
        self.x_max = 2.0    # Maximum cart position [m]
        self.u_max = 20.0   # Maximum force [N]
        
        # Control parameters
        self.angle_threshold = np.pi / 6  # 30 degrees - beyond this, LQR is invalid

params = PendulumParams()

# ============================================================================
# DYNAMICS
# ============================================================================

def nonlinear_dynamics(x, u, p):
    """
    Full nonlinear dynamics of inverted pendulum
    
    State x = [position, angle, velocity, angular_velocity]
    Input u = force on cart
    
    Returns: state derivatives [ẋ, θ̇, ẍ, θ̈]
    """
    pos, theta, vel, omega = x
    
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Denominator term appearing in both accelerations
    denom = p.M + p.m * sin_theta**2
    
    # Cart acceleration
    x_ddot = (u + p.m * sin_theta * (p.l * omega**2 + p.g * cos_theta) - p.b * vel) / denom
    
    # Pendulum angular acceleration
    theta_ddot = (
        -u * cos_theta 
        - p.m * p.l * omega**2 * cos_theta * sin_theta
        - (p.M + p.m) * p.g * sin_theta
        + p.b * vel * cos_theta
    ) / (p.l * denom)
    
    return np.array([vel, omega, x_ddot, theta_ddot])

def simulate_step(x, u, p):
    """Euler integration of nonlinear dynamics"""
    x_dot = nonlinear_dynamics(x, u, p)
    x_new = x + p.dt * x_dot
    
    # Enforce position constraints
    x_new[0] = np.clip(x_new[0], -p.x_max, p.x_max)
    if abs(x_new[0]) >= p.x_max:
        x_new[2] = 0.0  # Stop velocity at boundary
    
    return x_new

def get_linearized_matrices(p):
    """
    Linearize system around upright equilibrium (θ = 0)
    
    Returns continuous-time A, B matrices
    Then discretizes them
    """
    # Continuous-time linearization
    A_cont = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, -p.m * p.g / p.M, -p.b / p.M, 0.0],
        [0.0, (p.M + p.m) * p.g / (p.M * p.l), p.b / (p.M * p.l), 0.0]
    ])
    
    B_cont = np.array([
        [0.0],
        [0.0],
        [1.0 / p.M],
        [-1.0 / (p.M * p.l)]
    ])
    
    # Discretize using forward Euler (simple but effective for small dt)
    A_disc = np.eye(4) + p.dt * A_cont
    B_disc = p.dt * B_cont
    
    return A_disc, B_disc

# ============================================================================
# LQR CONTROLLER
# ============================================================================

class LQRController:
    """Linear Quadratic Regulator for stabilization near upright"""
    
    def __init__(self, Q, R, params):
        """
        Q: State cost matrix (4x4)
        R: Input cost matrix (1x1)
        """
        self.Q = Q
        self.R = R
        self.params = params
        self.K = None
        self.compute_gain()
        
    def compute_gain(self):
        """
        Solve Discrete Algebraic Riccati Equation (DARE) and compute optimal gain
        
        DARE: P = A^T P A - A^T P B (R + B^T P B)^-1 B^T P A + Q
        Optimal gain: K = (B^T P B + R)^-1 B^T P A
        """
        A, B = get_linearized_matrices(self.params)
        
        # Solve DARE using scipy (more robust than manual iteration)
        P = solve_discrete_are(A, B, self.Q, self.R)
        
        # Compute LQR gain
        self.K = np.linalg.solve(self.R + B.T @ P @ B, B.T @ P @ A)
        
        print("\n=== LQR Gain Matrix ===")
        print(f"K = {self.K[0]}")
        print(f"Control law: u = -K * [x, θ, ẋ, θ̇]^T")
        
    def control(self, x):
        """Compute control input u = -K * x"""
        # Check if angle is too large for LQR
        if abs(x[1]) > self.params.angle_threshold:
            # Switch to swing-up
            return self.swing_up_control(x)
        
        # Standard LQR control
        u = -self.K @ x
        return np.clip(u[0], -self.params.u_max, self.params.u_max)
    
    def swing_up_control(self, x):
        """
        Energy-based swing-up controller for large angles
        
        Pumps energy into the system to bring pendulum upright
        """
        pos, theta, vel, omega = x
        
        # Total energy of pendulum
        E = 0.5 * self.params.m * self.params.l**2 * omega**2 \
            - self.params.m * self.params.g * self.params.l * np.cos(theta)
        
        # Desired energy at upright position
        E_desired = self.params.m * self.params.g * self.params.l
        
        # Energy shaping control
        k_energy = 2.0
        k_damping = 0.5
        
        u = k_energy * (E - E_desired) * np.sign(omega * np.cos(theta)) - k_damping * vel
        
        return np.clip(u, -self.params.u_max, self.params.u_max)

# ============================================================================
# MPC CONTROLLER
# ============================================================================

class MPCController:
    """Model Predictive Control using full nonlinear model"""
    
    def __init__(self, Q_diag, R, N, params):
        """
        Q_diag: Diagonal elements of state cost matrix [q_x, q_theta, q_vel, q_omega]
        R: Input cost weight
        N: Prediction horizon
        """
        self.Q_diag = np.array(Q_diag)
        self.R = R
        self.N = N  # Prediction horizon
        self.params = params
        
        print("\n=== MPC Parameters ===")
        print(f"Prediction horizon: {N} steps ({N * params.dt:.2f} seconds)")
        print(f"State weights: {Q_diag}")
        print(f"Input weight: {R}")
        
    def cost_function(self, x0, u_seq):
        """
        Evaluate total cost over prediction horizon
        
        Cost = Σ(x^T Q x + u^T R u) + constraints penalties
        """
        x = x0.copy()
        total_cost = 0.0
        
        for k in range(self.N):
            # State cost
            state_cost = np.sum(self.Q_diag * x**2)
            
            # Input cost
            input_cost = self.R * u_seq[k]**2
            
            # Position constraint penalty (soft constraint)
            if abs(x[0]) > self.params.x_max:
                position_penalty = 1000 * (abs(x[0]) - self.params.x_max)**2
            else:
                position_penalty = 0.0
            
            total_cost += state_cost + input_cost + position_penalty
            
            # Simulate forward
            x = simulate_step(x, u_seq[k], self.params)
        
        return total_cost
    
    def control(self, x):
        """
        Solve optimization problem to find optimal control sequence
        
        Uses simple gradient descent (real MPC would use proper optimization)
        """
        # Initialize control sequence
        u_seq = np.zeros(self.N)
        
        # Gradient descent parameters
        learning_rate = 0.2
        n_iterations = 15
        
        for iteration in range(n_iterations):
            # Compute gradient via finite differences
            grad = np.zeros(self.N)
            eps = 0.01
            
            for k in range(self.N):
                # Cost with current control
                cost_curr = self.cost_function(x, u_seq)
                
                # Cost with perturbed control
                u_pert = u_seq.copy()
                u_pert[k] += eps
                cost_pert = self.cost_function(x, u_pert)
                
                # Finite difference gradient
                grad[k] = (cost_pert - cost_curr) / eps
            
            # Gradient descent update with projection onto constraints
            u_seq = u_seq - learning_rate * grad
            u_seq = np.clip(u_seq, -self.params.u_max, self.params.u_max)
        
        # Return first control input (receding horizon principle)
        return u_seq[0]

# ============================================================================
# SIMULATION AND VISUALIZATION
# ============================================================================

def simulate_controller(controller, x0, T, params, controller_name):
    """Run simulation with given controller"""
    
    x = x0.copy()
    t = 0.0
    
    # Storage for plotting
    time_hist = [0.0]
    state_hist = [x.copy()]
    control_hist = [0.0]
    
    n_steps = int(T / params.dt)
    
    print(f"\n=== Simulating {controller_name} ===")
    start_time = time.time()
    
    for step in range(n_steps):
        # Compute control
        u = controller.control(x)
        
        # Simulate
        x = simulate_step(x, u, params)
        t += params.dt
        
        # Store history
        time_hist.append(t)
        state_hist.append(x.copy())
        control_hist.append(u)
        
        # Print progress
        if step % 50 == 0:
            print(f"t={t:.2f}s: pos={x[0]:.3f}m, angle={np.rad2deg(x[1]):.2f}°, u={u:.2f}N")
    
    elapsed = time.time() - start_time
    print(f"Simulation completed in {elapsed:.3f}s")
    print(f"Final state: pos={x[0]:.3f}m, angle={np.rad2deg(x[1]):.2f}°")
    
    return np.array(time_hist), np.array(state_hist), np.array(control_hist)

def plot_comparison(results_dict):
    """Plot comparison of different controllers"""
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Inverted Pendulum Control Comparison: LQR vs MPC', fontsize=14, fontweight='bold')
    
    colors = {'LQR': 'blue', 'MPC': 'red'}
    
    for name, (t, states, controls) in results_dict.items():
        color = colors.get(name, 'black')
        
        # Position
        axes[0].plot(t, states[:, 0], label=name, color=color, linewidth=2)
        axes[0].axhline(y=params.x_max, color='red', linestyle='--', alpha=0.3, label='Constraint' if name == 'LQR' else '')
        axes[0].axhline(y=-params.x_max, color='red', linestyle='--', alpha=0.3)
        axes[0].set_ylabel('Position [m]', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='upper right')
        
        # Angle
        axes[1].plot(t, np.rad2deg(states[:, 1]), label=name, color=color, linewidth=2)
        axes[1].axhline(y=np.rad2deg(params.angle_threshold), color='orange', linestyle='--', alpha=0.3, label='LQR Valid' if name == 'LQR' else '')
        axes[1].axhline(y=-np.rad2deg(params.angle_threshold), color='orange', linestyle='--', alpha=0.3)
        axes[1].set_ylabel('Angle [deg]', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='upper right')
        
        # Control input
        axes[2].plot(t, controls, label=name, color=color, linewidth=2)
        axes[2].axhline(y=params.u_max, color='red', linestyle='--', alpha=0.3)
        axes[2].axhline(y=-params.u_max, color='red', linestyle='--', alpha=0.3)
        axes[2].set_ylabel('Control Force [N]', fontsize=11)
        axes[2].set_xlabel('Time [s]', fontsize=11)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('lqr_vs_mpc_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'lqr_vs_mpc_comparison.png'")
    plt.show()

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("INVERTED PENDULUM CONTROL: LQR vs MPC")
    print("="*70)
    
    # Initial condition: small angle perturbation
    x0 = np.array([0.0, 0.3, 0.0, 0.0])  # [pos, angle, vel, omega]
    print(f"\nInitial state: pos={x0[0]:.2f}m, angle={np.rad2deg(x0[1]):.2f}°")
    
    # Simulation time
    T_sim = 5.0
    
    # ========================================================================
    # LQR Setup
    # ========================================================================
    Q_lqr = np.diag([10.0, 100.0, 1.0, 1.0])  # Penalize angle heavily
    R_lqr = np.array([[1.0]])
    
    lqr = LQRController(Q_lqr, R_lqr, params)
    
    # ========================================================================
    # MPC Setup
    # ========================================================================
    Q_mpc = [10.0, 100.0, 1.0, 1.0]  # Same weights as LQR for fair comparison
    R_mpc = 1.0
    N_horizon = 20  # 20 steps * 0.02s = 0.4s lookahead
    
    mpc = MPCController(Q_mpc, R_mpc, N_horizon, params)
    
    # ========================================================================
    # Run Simulations
    # ========================================================================
    results = {}
    
    print("\n" + "-"*70)
    t_lqr, states_lqr, controls_lqr = simulate_controller(lqr, x0, T_sim, params, "LQR")
    results['LQR'] = (t_lqr, states_lqr, controls_lqr)
    
    print("\n" + "-"*70)
    t_mpc, states_mpc, controls_mpc = simulate_controller(mpc, x0, T_sim, params, "MPC")
    results['MPC'] = (t_mpc, states_mpc, controls_mpc)
    
    # ========================================================================
    # Visualize Results
    # ========================================================================
    plot_comparison(results)
    
    # ========================================================================
    # Performance Metrics
    # ========================================================================
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    
    for name, (t, states, controls) in results.items():
        position_error = np.mean(states[:, 0]**2)
        angle_error = np.mean(states[:, 1]**2)
        control_effort = np.mean(controls**2)
        max_position = np.max(np.abs(states[:, 0]))
        max_angle = np.rad2deg(np.max(np.abs(states[:, 1])))
        
        print(f"\n{name}:")
        print(f"  Mean squared position error: {position_error:.6f}")
        print(f"  Mean squared angle error: {angle_error:.6f}")
        print(f"  Mean squared control effort: {control_effort:.4f}")
        print(f"  Maximum position deviation: {max_position:.4f} m")
        print(f"  Maximum angle deviation: {max_angle:.2f} deg")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("="*70)
    print("""
    LQR (Linear Quadratic Regulator):
    - Extremely fast computation (just matrix multiplication)
    - Optimal for small deviations from equilibrium
    - Breaks down for large angles (linearization invalid)
    - No explicit constraint handling
    
    MPC (Model Predictive Control):
    - Uses full nonlinear dynamics
    - Explicitly handles state and input constraints
    - More robust to large disturbances
    - Computationally intensive (optimization at each step)
    - Can "see" future constraints and plan accordingly
    """)

if __name__ == '__main__':
    main()