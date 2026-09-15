import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
from matplotlib.offsetbox import AnchoredText

def plot_aligned_trajectories(csv_file='../../bags/09-03-27.csv', start_index=0, end_index=None):
    """
    Plots the aligned trajectories of Sonar Odometry and DVL and overlays 
    the mean translation and rotation errors in a text box.
    
    Parameters:
    - csv_file: Path to the CSV file containing the data.
    - start_index: The starting row index for the subsection (default: 0).
    - end_index: The ending row index for the subsection (default: None, goes to the end).
    """
    
    # 1. Load the data
    try:
        df = pd.read_csv(csv_file)
        print(f"Successfully loaded data from '{csv_file}'. Total rows: {len(df)}")
    except FileNotFoundError:
        print(f"Error: Could not find '{csv_file}'. Ensure the path is correct.")
        return

    # Slice the dataframe to grab the requested subsection
    df_subset = df.iloc[start_index:end_index].copy()
    
    if df_subset.empty:
        print("Error: The specified index range resulted in an empty dataset.")
        return

    # 2. Extract Sonar Odometry data and FLIP the Y-axis and Yaw
    odom_x = df_subset['sonar_odom_x'].values
    odom_y = -df_subset['sonar_odom_y'].values     # Flipped Y
    odom_yaw = -df_subset['sonar_odom_yaw'].values # Flipped Yaw to match the mirrored Y-axis

    # Extract DVL data
    dvl_x = df_subset['dvl_pos_x'].values
    dvl_y = df_subset['dvl_pos_y'].values
    dvl_yaw = df_subset['dvl_pos_yaw'].values

    # 3. Align the DVL path to the Odom path at the new starting timestep
    # Step A: Shift DVL so its first point of the subsection is at the origin (0, 0)
    dvl_x_shifted = dvl_x - dvl_x[0]
    dvl_y_shifted = dvl_y - dvl_y[0]

    # Step B: Calculate the initial yaw difference to align headings
    yaw_diff = odom_yaw[0] - dvl_yaw[0]
    cos_diff = np.cos(yaw_diff)
    sin_diff = np.sin(yaw_diff)

    # Step C: Rotate the DVL path by the yaw difference
    dvl_x_rotated = dvl_x_shifted * cos_diff - dvl_y_shifted * sin_diff
    dvl_y_rotated = dvl_x_shifted * sin_diff + dvl_y_shifted * cos_diff

    # Step D: Translate the rotated DVL path to start exactly at the Odom's start point
    dvl_x_aligned = dvl_x_rotated + odom_x[0]
    dvl_y_aligned = dvl_y_rotated + odom_y[0]

    # Step E: Align DVL yaw heading to match initial Odom yaw
    dvl_yaw_aligned = dvl_yaw + yaw_diff

    # 4. Compute the mean translation and rotation errors
    trans_error_2d = np.sqrt((odom_x - dvl_x_aligned)**2 + (odom_y - dvl_y_aligned)**2)
    mean_trans_error = np.mean(trans_error_2d)

    error_yaw = np.arctan2(np.sin(odom_yaw - dvl_yaw_aligned), 
                           np.cos(odom_yaw - dvl_yaw_aligned))
    mean_rot_error_rad = np.mean(np.abs(error_yaw))
    mean_rot_error_deg = np.rad2deg(mean_rot_error_rad)

    # Print to console as well
    print(f"Mean Translation Error: {mean_trans_error:.4f} m")
    print(f"Mean Rotation Error   : {mean_rot_error_rad:.4f} rad ({mean_rot_error_deg:.2f}°)")

    # 5. Setup the Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Set light figure background and slightly dark axes background for contrast
    fig.patch.set_facecolor('white')
    ax.set_facecolor("#777575")
    
    # Plot both paths 
    ax.plot(odom_x, odom_y, label='SOL-SLAM (FLS only; Ours)', 
            color='#91FF82', linestyle='-.', alpha=0.9, linewidth=2.5)
    ax.plot(dvl_x_aligned, dvl_y_aligned, label='DVL', 
            color='#FF3333', linestyle='-', alpha=0.9, linewidth=2.5)

    # Single white circle for the shared start position
    ax.scatter(odom_x[0], odom_y[0], color='w', marker='o', 
               s=100, edgecolor='black', zorder=6)

    # Mark the end points
    ax.scatter(odom_x[-1], odom_y[-1], color='#91FF82', marker='X', 
               s=80, edgecolor='black', zorder=5)
    ax.scatter(dvl_x_aligned[-1], dvl_y_aligned[-1], color='#FF3333', marker='X', 
               s=80, edgecolor='black', zorder=5)

    # Formatting the plot axes
    ax.set_xlabel('X (meters)', fontsize=12, color='black')
    ax.set_ylabel('Y (meters)', fontsize=12, color='black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    for spine in ax.spines.values():
        spine.set_edgecolor('black')

    # Custom main legend handles
    handles, labels = ax.get_legend_handles_labels()
    start_marker = mlines.Line2D([], [], color='w', marker='o', linestyle='None', 
                                 markersize=10, markeredgecolor='black', label='Start Point')
    end_marker = mlines.Line2D([], [], color='none', marker='X', linestyle='None', 
                               markersize=10, markeredgecolor='black', label='End Point')
    
    handles.extend([start_marker, end_marker])
    
    # Light mode styling for the legend
    legend = ax.legend(handles=handles, fontsize=11, loc='upper left', framealpha=1.0)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#cccccc')
    for text in legend.get_texts():
        text.set_color("black")

    # 6. Add Formatted Error Text Box (Styled based on plot_aracati.py)
    error_text = (
        "RMSE vs DVL\n"
        "─────────────────\n"
        f"Translation:   {mean_trans_error:.2f} m\n"
        f"Rotation:     {mean_rot_error_deg:.2f}°"
    )
    
    anchored_text = AnchoredText(
        error_text, 
        loc='lower right',
        borderpad=0.93,
        prop=dict(fontsize=11, multialignment='left'),
        frameon=True
    )
    
    # Match the legend box styling exactly
    anchored_text.patch.set_boxstyle("round,pad=0.4")
    anchored_text.patch.set_facecolor("white")
    anchored_text.patch.set_edgecolor("#cccccc")
    anchored_text.patch.set_linewidth(1.0)
    anchored_text.patch.set_alpha(1.0)
    anchored_text.set_zorder(10)
    
    ax.add_artist(anchored_text)

    # Grid and layout
    ax.grid(True, linestyle=':', alpha=0.4, color="#ffffff")
    ax.axis('equal') # Ensure 1 meter in X looks the same as 1 meter in Y
    plt.tight_layout()

    # Display the plot
    plt.savefig(f"seq_3.pdf", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    #plot_aligned_trajectories(start_index=100, end_index=350)
    #plot_aligned_trajectories(start_index=350, end_index=550)
    plot_aligned_trajectories(start_index=700, end_index=900)

    '''
    13_14-42-05
    plot_aligned_trajectories(start_index=50, end_index=300)

    13_14-44-40
    plot_aligned_trajectories(start_index=200, end_index=300)
    plot_aligned_trajectories(start_index=500, end_index=None)

    13_14-52-03
    plot_aligned_trajectories(start_index=0, end_index=100)

    09-03-27
    plot_aligned_trajectories(start_index=100, end_index=350)
    plot_aligned_trajectories(start_index=350, end_index=550)
    plot_aligned_trajectories(start_index=700, end_index=900)
    '''