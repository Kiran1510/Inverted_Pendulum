"""
Inverted Pendulum Control: LQR (Linear Quadratic Regulator)
Interactive visualization with real-time control
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
from matplotlib.widgets import Slider
from scipy.linalg import solve_discrete_are

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
        self.x_max = 3.0    # Maximum cart position [m]
        self.u_max = 20.0   # Maximum force [N]
        
        # Control parameters
        self.angle_threshold = np.pi / 6  # 30 degrees - beyond this, LQR is invalid

params = PendulumParams()

# ============================================================================
# DYNAMICS
# ============================================================================

def nonlinear_dynamics(x, u, p):
    """Full nonlinear dynamics of inverted pendulum"""
    pos, theta, vel, omega = x
    
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    denom = p.M + p.m * sin_theta**2
    
    x_ddot = (u + p.m * sin_theta * (p.l * omega**2 + p.g * cos_theta) - p.b * vel) / denom
    
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
        x_new[2] = 0.0
    
    return x_new

def get_linearized_matrices(p):
    """Linearize system around upright equilibrium"""
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
    
    A_disc = np.eye(4) + p.dt * A_cont
    B_disc = p.dt * B_cont
    
    return A_disc, B_disc

# ============================================================================
# LQR CONTROLLER
# ============================================================================

class LQRController:
    """Linear Quadratic Regulator for stabilization near upright"""
    
    def __init__(self, Q, R, params):
        self.Q = Q
        self.R = R
        self.params = params
        self.K = None
        self.compute_gain()
        
    def compute_gain(self):
        """Solve DARE and compute optimal gain"""
        A, B = get_linearized_matrices(self.params)
        P = solve_discrete_are(A, B, self.Q, self.R)
        self.K = np.linalg.solve(self.R + B.T @ P @ B, B.T @ P @ A)
        
        print("\n=== LQR Gain Matrix ===")
        print(f"K = {self.K[0]}")
        print(f"Control law: u = -K * [x, θ, ẋ, θ̇]^T")
        
    def control(self, x):
        """Compute control input with swing-up fallback"""
        if abs(x[1]) > self.params.angle_threshold:
            return self.swing_up_control(x)
        
        u = -self.K @ x
        return np.clip(u[0], -self.params.u_max, self.params.u_max)
    
    def swing_up_control(self, x):
        """Energy-based swing-up controller"""
        pos, theta, vel, omega = x
        
        E = 0.5 * self.params.m * self.params.l**2 * omega**2 \
            - self.params.m * self.params.g * self.params.l * np.cos(theta)
        
        E_desired = self.params.m * self.params.g * self.params.l
        
        k_energy = 2.0
        k_damping = 0.5
        
        u = k_energy * (E - E_desired) * np.sign(omega * np.cos(theta)) - k_damping * vel
        
        return np.clip(u, -self.params.u_max, self.params.u_max)

# ============================================================================
# INTERACTIVE VISUALIZATION
# ============================================================================

class InteractiveSimulation:
    """Real-time animated visualization of cart-pole control"""
    
    def __init__(self, controller, x0, params):
        self.controller = controller
        self.x = x0.copy()
        self.params = params
        
        # Disturbance parameters
        self.disturbance_force = 0.0
        self.disturbance_angle = 0.0
        
        # History for plotting
        self.time_hist = []
        self.pos_hist = []
        self.angle_hist = []
        self.control_hist = []
        self.t = 0.0
        
        # Setup figure with extra space for controls
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.suptitle('Inverted Pendulum - LQR Control', 
                         fontsize=14, fontweight='bold')
        
        # Create subplots
        gs = self.fig.add_gridspec(3, 2, height_ratios=[2, 1, 0.3], hspace=0.35, wspace=0.3)
        
        # Cart-pole animation
        self.ax_cart = self.fig.add_subplot(gs[0, :])
        self.ax_cart.set_xlim(-4, 4)
        self.ax_cart.set_ylim(-0.5, 1.5)
        self.ax_cart.set_aspect('equal')
        self.ax_cart.set_xlabel('Position [m]')
        self.ax_cart.grid(True, alpha=0.3)
        
        # Position plot
        self.ax_pos = self.fig.add_subplot(gs[1, 0])
        self.ax_pos.set_xlim(0, 20)
        self.ax_pos.set_ylim(-3.5, 3.5)
        self.ax_pos.set_ylabel('Position [m]')
        self.ax_pos.set_xlabel('Time [s]')
        self.ax_pos.grid(True, alpha=0.3)
        self.ax_pos.axhline(y=params.x_max, color='red', linestyle='--', alpha=0.3)
        self.ax_pos.axhline(y=-params.x_max, color='red', linestyle='--', alpha=0.3)
        
        # Angle plot
        self.ax_angle = self.fig.add_subplot(gs[1, 1])
        self.ax_angle.set_xlim(0, 20)
        self.ax_angle.set_ylim(-50, 50)
        self.ax_angle.set_ylabel('Angle [deg]')
        self.ax_angle.set_xlabel('Time [s]')
        self.ax_angle.grid(True, alpha=0.3)
        self.ax_angle.axhline(y=np.rad2deg(params.angle_threshold), 
                             color='orange', linestyle='--', alpha=0.3, label='LQR Valid Zone')
        self.ax_angle.axhline(y=-np.rad2deg(params.angle_threshold), 
                             color='orange', linestyle='--', alpha=0.3)
        self.ax_angle.legend(loc='upper right', fontsize=8)
        
        # Initialize plot elements
        self.pos_line, = self.ax_pos.plot([], [], 'b-', linewidth=2)
        self.angle_line, = self.ax_angle.plot([], [], 'r-', linewidth=2)
        
        # Cart-pole elements
        self.cart_width = 0.3
        self.cart_height = 0.15
        self.wheel_radius = 0.04
        
        self.cart_rect = Rectangle((0, 0), self.cart_width, self.cart_height, 
                                   fc='blue', ec='darkblue', linewidth=2, zorder=2)
        self.ax_cart.add_patch(self.cart_rect)
        
        self.wheel_left = Circle((0, 0), self.wheel_radius, fc='black', zorder=2)
        self.wheel_right = Circle((0, 0), self.wheel_radius, fc='black', zorder=2)
        self.ax_cart.add_patch(self.wheel_left)
        self.ax_cart.add_patch(self.wheel_right)
        
        self.pendulum_line, = self.ax_cart.plot([], [], 'r-', linewidth=4, zorder=3)
        self.pendulum_bob = Circle((0, 0), 0.08, fc='darkred', ec='black', linewidth=2, zorder=4)
        self.ax_cart.add_patch(self.pendulum_bob)
        
        # Ground line to cover artifacts
        self.ax_cart.axhline(y=0, color='gray', linewidth=3, zorder=1)
        
        # Track limits
        self.ax_cart.axvline(x=params.x_max, color='red', linestyle='--', linewidth=2, alpha=0.5)
        self.ax_cart.axvline(x=-params.x_max, color='red', linestyle='--', linewidth=2, alpha=0.5)
        
        # Info text
        self.info_text = self.ax_cart.text(0.02, 0.98, '', transform=self.ax_cart.transAxes,
                                           verticalalignment='top', fontfamily='monospace',
                                           fontsize=10, bbox=dict(boxstyle='round', 
                                           facecolor='wheat', alpha=0.8))
        
        # =================================================================
        # Interactive Controls
        # =================================================================
        
        # Force disturbance slider
        ax_force = self.fig.add_subplot(gs[2, 0])
        self.force_slider = Slider(
            ax=ax_force,
            label='Force Disturbance [N]',
            valmin=-10.0,
            valmax=10.0,
            valinit=0.0,
            valstep=0.5
        )
        self.force_slider.on_changed(self.update_force_disturbance)
        
        # Angle disturbance slider
        ax_angle = self.fig.add_subplot(gs[2, 1])
        self.angle_slider = Slider(
            ax=ax_angle,
            label='Angle Kick [deg]',
            valmin=-30.0,
            valmax=30.0,
            valinit=0.0,
            valstep=1.0
        )
        self.angle_slider.on_changed(self.update_angle_disturbance)
        
    def update_force_disturbance(self, val):
        """Update force disturbance from slider"""
        self.disturbance_force = val
        
    def update_angle_disturbance(self, val):
        """Update angle disturbance and apply it immediately"""
        # Apply angle kick to pendulum
        self.x[1] += np.deg2rad(val - self.disturbance_angle)
        self.disturbance_angle = val
    
    def init_animation(self):
        """Initialize animation"""
        self.pos_line.set_data([], [])
        self.angle_line.set_data([], [])
        self.pendulum_line.set_data([], [])
        
        # Set initial cart and pendulum positions
        cart_x = self.x[0] - self.cart_width / 2
        self.cart_rect.set_xy((cart_x, 0))
        
        pend_x = self.x[0] + self.params.l * np.sin(self.x[1])
        pend_y = self.cart_height + self.params.l * np.cos(self.x[1])
        self.pendulum_line.set_data([self.x[0], pend_x], [self.cart_height, pend_y])
        self.pendulum_bob.center = (pend_x, pend_y)
        
        return self.pos_line, self.angle_line, self.pendulum_line, self.cart_rect, self.pendulum_bob
    
    def update(self, frame):
        """Update function for animation"""
        # Compute control
        u = self.controller.control(self.x)
        
        # Add force disturbance
        u_total = u + self.disturbance_force
        
        # Simulate
        self.x = simulate_step(self.x, u_total, self.params)
        self.t += self.params.dt
        
        # Store history
        self.time_hist.append(self.t)
        self.pos_hist.append(self.x[0])
        self.angle_hist.append(np.rad2deg(self.x[1]))
        self.control_hist.append(u)
        
        # Keep history manageable
        max_hist = 1000
        if len(self.time_hist) > max_hist:
            self.time_hist.pop(0)
            self.pos_hist.pop(0)
            self.angle_hist.pop(0)
            self.control_hist.pop(0)
        
        # Update cart position
        cart_x = self.x[0] - self.cart_width / 2
        self.cart_rect.set_xy((cart_x, 0))
        
        # Update wheels
        self.wheel_left.center = (self.x[0] - self.cart_width/4, self.wheel_radius)
        self.wheel_right.center = (self.x[0] + self.cart_width/4, self.wheel_radius)
        
        # Update pendulum
        pend_x = self.x[0] + self.params.l * np.sin(self.x[1])
        pend_y = self.cart_height + self.params.l * np.cos(self.x[1])
        self.pendulum_line.set_data([self.x[0], pend_x], 
                                    [self.cart_height, pend_y])
        self.pendulum_bob.center = (pend_x, pend_y)
        
        # Change cart color based on constraint violation
        if abs(self.x[0]) >= self.params.x_max * 0.95:
            self.cart_rect.set_facecolor('red')
        elif abs(self.x[0]) >= self.params.x_max * 0.8:
            self.cart_rect.set_facecolor('orange')
        else:
            self.cart_rect.set_facecolor('blue')
        
        # Update plots
        if len(self.time_hist) > 1:
            self.pos_line.set_data(self.time_hist, self.pos_hist)
            self.angle_line.set_data(self.time_hist, self.angle_hist)
            
            # Auto-scale x-axis for scrolling window
            if self.t > 20:
                self.ax_pos.set_xlim(self.t - 20, self.t)
                self.ax_angle.set_xlim(self.t - 20, self.t)
        
        # Update info text
        mode = ""
        if abs(self.x[1]) > self.params.angle_threshold:
            mode = " (SWING-UP MODE)"
        
        info_str = f"Time: {self.t:.2f}s{mode}\n"
        info_str += f"Position: {self.x[0]:.3f} m\n"
        info_str += f"Angle: {np.rad2deg(self.x[1]):.2f}°\n"
        info_str += f"Control: {u:.2f} N\n"
        if abs(self.disturbance_force) > 0.1:
            info_str += f"Force Dist: {self.disturbance_force:.1f} N"
        self.info_text.set_text(info_str)
        
        return (self.cart_rect, self.wheel_left, self.wheel_right, 
                self.pendulum_line, self.pendulum_bob, 
                self.pos_line, self.angle_line, self.info_text)
    
    def run(self, duration=20.0):
        """Run the interactive animation"""
        n_frames = int(duration / self.params.dt)
        
        anim = FuncAnimation(self.fig, self.update, 
                           init_func=self.init_animation,
                           frames=n_frames, 
                           interval=self.params.dt * 1000,  # milliseconds
                           blit=True, repeat=False)
        
        plt.show()
        
        return anim

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("INVERTED PENDULUM CONTROL: LQR (Linear Quadratic Regulator)")
    print("="*70)
    
    # Get initial angle from user
    print("\nInitial Conditions:")
    angle_input = input("Enter initial angle in degrees (default=0): ").strip()
    
    if angle_input == "":
        angle_deg = 0.0
        print("Using default: 0 degrees")
    else:
        try:
            angle_deg = float(angle_input)
        except ValueError:
            print("Invalid input, using default: 0 degrees")
            angle_deg = 0.0
    
    # Initial condition
    x0 = np.array([0.0, np.deg2rad(angle_deg), 0.0, 0.0])  # [pos, angle, vel, omega]
    print(f"Initial state: pos={x0[0]:.2f}m, angle={angle_deg:.2f}°")
    
    # LQR Setup
    print("\nInitializing LQR Controller...")
    Q_lqr = np.diag([10.0, 100.0, 1.0, 1.0])  # Penalize angle heavily
    R_lqr = np.array([[1.0]])
    
    controller = LQRController(Q_lqr, R_lqr, params)
    
    # Run interactive simulation
    print("\n" + "="*70)
    print("Starting interactive simulation...")
    print("Use sliders to apply disturbances")
    print("Close the window to exit")
    print("="*70 + "\n")
    
    sim = InteractiveSimulation(controller, x0, params)
    sim.run(duration=20.0)
    
    print("\nSimulation complete!")

if __name__ == '__main__':
    main()