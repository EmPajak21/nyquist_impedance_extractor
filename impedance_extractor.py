import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Uncomment for Latex Graph Formatting
plt.rcParams.update({
    'text.usetex': True,  
    'font.size': 15,
    'axes.titlesize': 16, 
    'axes.labelsize': 15,
    'xtick.labelsize': 15, 
    'ytick.labelsize': 15,
    'legend.fontsize': 15 
})

# Import experimental data from an Excel file
file_name = input("Please input file name:")
df = pd.read_excel(file_name).iloc[:, [2, 3]] 
df = pd.DataFrame({"x": df.iloc[:, 0], "y": df.iloc[:, 1]}) 


def zero_grad(df, tol=1e-5, index=False, shift=8):
    """
    Find the point where the gradient is approximately zero in the given experimental data.
    
    Parameters:
    - df (DataFrame): DataFrame with 'x' and 'y' columns.
    - tol (float): Tolerance for determining zero gradient.
    - index (bool): Whether to return the index of the point or the (x, y) coordinates.
    - shift (int): Shift value to adjust the index.

    Returns:
    - int/tuple: Index of zero gradient or the (x, y) coordinates, depending on 'index'.
    """
    # Compute gradient using numpy
    dx = df["x"].iloc[1] - df["x"].iloc[0]  
    dy = np.gradient(df["y"], dx) 

    # Find indices where the gradient is approximately zero (with tolerance)
    zero_indices = np.where(np.abs(dy) < tol)[0]
    zero_indices = [idx for idx in zero_indices if idx > 25]  # Ignore early data points

    # Return either index or (x, y) coordinates based on the index flag
    if not zero_indices and tol < 10:
        # Recursively increase tolerance if no zero gradients are found
        return zero_grad(df, tol * 1.2, index)

    if index:
        return zero_indices[0] - shift
    else:
        return (df["x"].iloc[zero_indices[0]], df["y"].iloc[zero_indices[0]]) if zero_indices else None


def circle_equation(params, x, y):
    """
    Compute the circle equation given parameters and data.
    
    Parameters:
    - params (list/tuple): Circle parameters (xc, yc, R).
    - x (array-like): x-coordinates.
    - y (array-like): y-coordinates.
    """
    xc, yc, R = params
    return (x - xc) ** 2 + (y - yc) ** 2 - R**2


def objective_function(params, x, y):
    """
    Objective function to minimize for fitting a circle.

    Returns:
    - float: Sum of squared differences between data and circle.
    """
    return np.sum(circle_equation(params, x, y) ** 2)


def get_result_graph(result, x_data, y_data):
    """
    Plot the experimental data and the fitted circle.
    """
    xc_opt, yc_opt, R_opt = result.x

    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(8, 4))  # Create a new figure with given size

    # Plot scatter data for experimental results
    ax.scatter(x_data, y_data, color='navy', s=40, alpha=0.7, label='Experimental Data') 

    # Add the fitted circle to the plot
    circle = plt.Circle((xc_opt, yc_opt), R_opt, color='crimson', fill=False, linestyle='--', linewidth=2, label='Fitted Circle')
    ax.add_patch(circle)

    # Set labels and axis limits
    ax.text(x_data.iloc[-1]/2, y_data.iloc[-1]/2, f'Impedance: {R_opt:.0f} $\Omega$',bbox=dict(facecolor='white', alpha=0.5))
    ax.set_xlabel(r"$Z'  [\Omega]$")
    ax.set_ylabel(r"$Z'' [\Omega]$")
    ax.legend(loc='upper center', bbox_to_anchor=(0.45, 1.15), ncol=5, frameon=False)  

    # Set axis limits with a slight buffer
    ax.set_xlim([0, max(x_data) * 1.1])
    ax.set_ylim([0, max(y_data) * 1.1])

    # Adjust tick positions and add a grid
    ax.xaxis.set_ticks_position('bottom')  
    ax.yaxis.set_ticks_position('left') 
    ax.grid(True, linestyle='--', alpha=0.5) 
    
    # Ensure an equal aspect ratio
    ax.set_aspect('equal', 'box') 
    fig.tight_layout() 

    # Save the plot as a high-quality PNG with high DPI
    output_filename = "res_plot.png"
    fig.savefig(output_filename, dpi=300, bbox_inches='tight') 
    plt.show()


def get_impedance_eq(df, default_z_cross=True, graph=False):
    """
    Calculate the impedance using the zero gradient point and circle fitting.
    
    Parameters:
    - df (DataFrame): DataFrame with 'x' and 'y' columns.
    - default_z_cross (bool): Whether to use the default method to find zero gradient.
    - graph (bool): Whether to plot the graph with fitted circle.
    
    Returns:
    - float: Estimated impedance value.
    """
    if default_z_cross:
        z_cross = zero_grad(df, tol=1e-7, index=True)
    else:
        z_cross = -24
    
    x_data = df['x'].iloc[:z_cross]
    y_data = df['y'].iloc[:z_cross]

    # Initial guess for circle fitting and perform minimization
    params_guess = [np.mean(x_data), np.mean(y_data), (max(x_data) - min(x_data)) / 2]
    result = minimize(objective_function, params_guess, args=(x_data, y_data))
    
    if graph:
        get_result_graph(result, x_data, y_data)

    R_opt = result.x[2]  # Extract the optimized radius
    return R_opt * 2 


def get_final_impedance(df, graph=False):
    """
    Get the final impedance based on data analysis and circle fitting.
    
    Parameters:
    - df (DataFrame): DataFrame with 'x' and 'y' columns.
    - graph (bool): Whether to plot the graph with fitted circle.
    
    Returns:
    - float: Final estimated impedance value.
    """
    # Get impedance based on default zero-crossing
    res = get_impedance_eq(df, graph=graph)
    
    if res > 1000:  # If impedance is high, use a different method
        D_opt = get_impedance_eq(df, default_z_cross=False)
        return D_opt
    else:
        return res


# Calculate and print the final impedance value
impedance = get_final_impedance(df, True)
print(f"Impedance: {impedance:.0f} Ω")
