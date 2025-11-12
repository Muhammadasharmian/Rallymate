"""
Generate comprehensive visualizations from analysis CSV data.
This script creates multiple plots and charts for datathon presentation.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_latest_csv():
    """Load the most recent CSV analysis file."""
    csv_dir = Path("output/analysisCSV")
    csv_files = list(csv_dir.glob("*.csv"))
    if not csv_files:
        print("Error: No CSV files found!")
        return None
    
    # Get the most recent file
    latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading data from: {latest_csv}")
    
    df = pd.read_csv(latest_csv)
    print(f"Loaded {len(df)} shots")
    return df


def create_shot_distribution_pie(df, output_dir):
    """Create pie chart showing shot type distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Overall distribution
    shot_counts = df['Shot_Type'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4']
    ax1.pie(shot_counts.values, labels=shot_counts.index, autopct='%1.1f%%',
            startangle=90, colors=colors, textprops={'fontsize': 14})
    ax1.set_title('Overall Shot Type Distribution', fontsize=16, fontweight='bold')
    
    # Per player distribution
    player_shot_counts = df.groupby(['Player', 'Shot_Type']).size().unstack(fill_value=0)
    player_shot_counts.plot(kind='bar', ax=ax2, color=colors, width=0.7)
    ax2.set_title('Shot Type Distribution by Player', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Player', fontsize=12)
    ax2.set_ylabel('Number of Shots', fontsize=12)
    ax2.legend(title='Shot Type', fontsize=12)
    ax2.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / '01_shot_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Created: 01_shot_distribution.png")
    plt.close()


def create_velocity_analysis(df, output_dir):
    """Create velocity analysis charts."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Velocity distribution histogram
    for player in df['Player'].unique():
        player_data = df[df['Player'] == player]['Max_Racket_Velocity_px_s']
        ax1.hist(player_data, alpha=0.6, bins=30, label=player, edgecolor='black')
    ax1.set_xlabel('Max Racket Velocity (px/s)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Max Racket Velocity', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Velocity by shot type
    shot_velocity = df.groupby('Shot_Type')['Max_Racket_Velocity_px_s'].apply(list)
    positions = range(len(shot_velocity))
    bp = ax2.boxplot(shot_velocity.values, positions=positions, labels=shot_velocity.index,
                      patch_artist=True, showmeans=True)
    for patch, color in zip(bp['boxes'], ['#FF6B6B', '#4ECDC4']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel('Max Racket Velocity (px/s)', fontsize=12)
    ax2.set_title('Velocity Distribution by Shot Type', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Velocity over time (shot progression)
    for player in df['Player'].unique():
        player_data = df[df['Player'] == player]
        ax3.plot(range(len(player_data)), player_data['Max_Racket_Velocity_px_s'], 
                marker='o', alpha=0.6, label=player, linewidth=2)
    ax3.set_xlabel('Shot Number', fontsize=12)
    ax3.set_ylabel('Max Racket Velocity (px/s)', fontsize=12)
    ax3.set_title('Velocity Progression Throughout Game', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Average velocity by player and shot type
    avg_velocity = df.groupby(['Player', 'Shot_Type'])['Max_Racket_Velocity_px_s'].mean().unstack()
    avg_velocity.plot(kind='bar', ax=ax4, color=['#FF6B6B', '#4ECDC4'], width=0.7)
    ax4.set_ylabel('Average Max Velocity (px/s)', fontsize=12)
    ax4.set_xlabel('Player', fontsize=12)
    ax4.set_title('Average Velocity by Player and Shot Type', fontsize=14, fontweight='bold')
    ax4.legend(title='Shot Type', fontsize=11)
    ax4.tick_params(axis='x', rotation=0)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / '02_velocity_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Created: 02_velocity_analysis.png")
    plt.close()


def create_angle_analysis(df, output_dir):
    """Create body angle analysis charts."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Hip/Knee angle distribution
    for player in df['Player'].unique():
        player_data = df[df['Player'] == player]['Min_Hip_Knee_Angle_deg']
        ax1.hist(player_data, alpha=0.6, bins=25, label=player, edgecolor='black')
    ax1.set_xlabel('Min Hip/Knee Angle (degrees)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Hip/Knee Angles', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Elbow angle distribution
    for player in df['Player'].unique():
        player_data = df[df['Player'] == player]['Min_Elbow_Angle_deg']
        ax2.hist(player_data, alpha=0.6, bins=25, label=player, edgecolor='black')
    ax2.set_xlabel('Min Elbow Angle (degrees)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Elbow Angles', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Angle comparison by shot type
    angles_data = [df[df['Shot_Type'] == 'Forehand']['Min_Hip_Knee_Angle_deg'],
                   df[df['Shot_Type'] == 'Backhand']['Min_Hip_Knee_Angle_deg']]
    bp1 = ax3.boxplot(angles_data, labels=['Forehand', 'Backhand'],
                       patch_artist=True, showmeans=True)
    for patch, color in zip(bp1['boxes'], ['#FF6B6B', '#4ECDC4']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_ylabel('Min Hip/Knee Angle (degrees)', fontsize=12)
    ax3.set_title('Hip/Knee Angles by Shot Type', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Elbow angle by shot type
    elbow_data = [df[df['Shot_Type'] == 'Forehand']['Min_Elbow_Angle_deg'],
                  df[df['Shot_Type'] == 'Backhand']['Min_Elbow_Angle_deg']]
    bp2 = ax4.boxplot(elbow_data, labels=['Forehand', 'Backhand'],
                       patch_artist=True, showmeans=True)
    for patch, color in zip(bp2['boxes'], ['#FF6B6B', '#4ECDC4']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.set_ylabel('Min Elbow Angle (degrees)', fontsize=12)
    ax4.set_title('Elbow Angles by Shot Type', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / '03_angle_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Created: 03_angle_analysis.png")
    plt.close()


def create_shot_duration_analysis(df, output_dir):
    """Create shot duration analysis."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Duration distribution
    for player in df['Player'].unique():
        player_data = df[df['Player'] == player]['Duration_Seconds']
        ax1.hist(player_data, alpha=0.6, bins=20, label=player, edgecolor='black')
    ax1.set_xlabel('Shot Duration (seconds)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Shot Duration', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Duration by shot type
    duration_data = [df[df['Shot_Type'] == 'Forehand']['Duration_Seconds'],
                     df[df['Shot_Type'] == 'Backhand']['Duration_Seconds']]
    bp = ax2.boxplot(duration_data, labels=['Forehand', 'Backhand'],
                     patch_artist=True, showmeans=True)
    for patch, color in zip(bp['boxes'], ['#FF6B6B', '#4ECDC4']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel('Duration (seconds)', fontsize=12)
    ax2.set_title('Shot Duration by Type', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Average duration comparison
    avg_duration = df.groupby(['Player', 'Shot_Type'])['Duration_Seconds'].mean().unstack()
    avg_duration.plot(kind='bar', ax=ax3, color=['#FF6B6B', '#4ECDC4'], width=0.7)
    ax3.set_ylabel('Average Duration (seconds)', fontsize=12)
    ax3.set_xlabel('Player', fontsize=12)
    ax3.set_title('Average Shot Duration by Player and Type', fontsize=14, fontweight='bold')
    ax3.legend(title='Shot Type', fontsize=11)
    ax3.tick_params(axis='x', rotation=0)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Duration vs Velocity scatter
    colors_map = {'Forehand': '#FF6B6B', 'Backhand': '#4ECDC4'}
    for shot_type in df['Shot_Type'].unique():
        shot_data = df[df['Shot_Type'] == shot_type]
        ax4.scatter(shot_data['Duration_Seconds'], shot_data['Max_Racket_Velocity_px_s'],
                   alpha=0.6, s=50, label=shot_type, color=colors_map[shot_type])
    ax4.set_xlabel('Duration (seconds)', fontsize=12)
    ax4.set_ylabel('Max Racket Velocity (px/s)', fontsize=12)
    ax4.set_title('Shot Duration vs Velocity', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '04_duration_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Created: 04_duration_analysis.png")
    plt.close()


def create_player_comparison(df, output_dir):
    """Create comprehensive player comparison."""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Total shots per player
    ax1 = fig.add_subplot(gs[0, 0])
    shot_counts = df['Player'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4']
    ax1.bar(shot_counts.index, shot_counts.values, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Number of Shots', fontsize=11)
    ax1.set_title('Total Shots per Player', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Average velocity comparison
    ax2 = fig.add_subplot(gs[0, 1])
    avg_velocity = df.groupby('Player')['Max_Racket_Velocity_px_s'].mean()
    ax2.bar(avg_velocity.index, avg_velocity.values, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Avg Velocity (px/s)', fontsize=11)
    ax2.set_title('Average Racket Velocity', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Average hip/knee angle
    ax3 = fig.add_subplot(gs[0, 2])
    avg_hip = df.groupby('Player')['Min_Hip_Knee_Angle_deg'].mean()
    ax3.bar(avg_hip.index, avg_hip.values, color=colors, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Avg Hip/Knee Angle (°)', fontsize=11)
    ax3.set_title('Average Hip/Knee Angle', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Performance radar chart
    ax4 = fig.add_subplot(gs[1, :], projection='polar')
    
    # Normalize metrics for radar chart
    metrics = ['Max_Racket_Velocity_px_s', 'Min_Hip_Knee_Angle_deg', 'Min_Elbow_Angle_deg', 'CoG_Horizontal_Movement_px']
    metric_labels = ['Velocity', 'Hip/Knee', 'Elbow', 'CoG Movement']
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    for i, player in enumerate(df['Player'].unique()):
        player_data = df[df['Player'] == player]
        values = []
        for metric in metrics:
            # Normalize to 0-1 range
            all_values = df[metric]
            player_value = player_data[metric].mean()
            normalized = (player_value - all_values.min()) / (all_values.max() - all_values.min())
            values.append(normalized)
        values += values[:1]
        
        ax4.plot(angles, values, 'o-', linewidth=2, label=player, color=colors[i])
        ax4.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(metric_labels)
    ax4.set_ylim(0, 1)
    ax4.set_title('Player Performance Comparison (Normalized)', fontsize=14, fontweight='bold', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax4.grid(True)
    
    # 5. Shot type breakdown
    ax5 = fig.add_subplot(gs[2, 0])
    shot_type_counts = df.groupby(['Player', 'Shot_Type']).size().unstack()
    shot_type_counts.plot(kind='bar', ax=ax5, color=['#FF6B6B', '#4ECDC4'], 
                          width=0.7, edgecolor='black', linewidth=1.5)
    ax5.set_ylabel('Number of Shots', fontsize=11)
    ax5.set_xlabel('Player', fontsize=11)
    ax5.set_title('Shot Type Breakdown', fontsize=12, fontweight='bold')
    ax5.legend(title='Shot Type', fontsize=10)
    ax5.tick_params(axis='x', rotation=0)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. CoG movement comparison
    ax6 = fig.add_subplot(gs[2, 1])
    avg_cog = df.groupby('Player')['CoG_Horizontal_Movement_px'].mean()
    ax6.bar(avg_cog.index, avg_cog.values, color=colors, edgecolor='black', linewidth=1.5)
    ax6.set_ylabel('Avg CoG Movement (px)', fontsize=11)
    ax6.set_title('Center of Gravity Movement', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Performance consistency (std deviation)
    ax7 = fig.add_subplot(gs[2, 2])
    velocity_std = df.groupby('Player')['Max_Racket_Velocity_px_s'].std()
    ax7.bar(velocity_std.index, velocity_std.values, color=colors, edgecolor='black', linewidth=1.5)
    ax7.set_ylabel('Velocity Std Dev', fontsize=11)
    ax7.set_title('Velocity Consistency\n(Lower = More Consistent)', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(output_dir / '05_player_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Created: 05_player_comparison.png")
    plt.close()


def create_correlation_heatmap(df, output_dir):
    """Create correlation heatmap of metrics."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Select numeric columns for correlation
    numeric_cols = ['Max_Racket_Velocity_px_s', 'Min_Hip_Knee_Angle_deg', 'Min_Elbow_Angle_deg', 
                    'CoG_Horizontal_Movement_px', 'Duration_Seconds']
    
    correlation = df[numeric_cols].corr()
    
    # Create heatmap
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax, vmin=-1, vmax=1)
    
    ax.set_title('Correlation Matrix of Biomechanical Metrics', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Better labels
    labels = ['Velocity', 'Hip/Knee\nAngle', 'Elbow\nAngle', 'CoG\nMovement', 'Duration']
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels, rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / '06_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Created: 06_correlation_heatmap.png")
    plt.close()


def create_timeline_analysis(df, output_dir):
    """Create timeline-based analysis."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Add shot number for timeline
    df_sorted = df.sort_values(['Player', 'Start_Frame']).reset_index(drop=True)
    
    # 1. Velocity over time
    for player in df_sorted['Player'].unique():
        player_data = df_sorted[df_sorted['Player'] == player].reset_index(drop=True)
        ax1.plot(player_data.index, player_data['Max_Racket_Velocity_px_s'], 
                marker='o', alpha=0.7, label=player, linewidth=2, markersize=4)
    ax1.set_xlabel('Shot Sequence', fontsize=12)
    ax1.set_ylabel('Max Racket Velocity (px/s)', fontsize=12)
    ax1.set_title('Velocity Progression Over Match', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Moving average of velocity
    window = 5
    for player in df_sorted['Player'].unique():
        player_data = df_sorted[df_sorted['Player'] == player].reset_index(drop=True)
        ma = player_data['Max_Racket_Velocity_px_s'].rolling(window=window, min_periods=1).mean()
        ax2.plot(player_data.index, ma, linewidth=3, label=f'{player} (MA-{window})', alpha=0.8)
    ax2.set_xlabel('Shot Sequence', fontsize=12)
    ax2.set_ylabel('Velocity Moving Average (px/s)', fontsize=12)
    ax2.set_title(f'Velocity Trend (Moving Avg, window={window})', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Shot type alternation
    colors_map = {'Forehand': '#FF6B6B', 'Backhand': '#4ECDC4'}
    for player in df_sorted['Player'].unique():
        player_data = df_sorted[df_sorted['Player'] == player].reset_index(drop=True)
        shot_colors = [colors_map[shot] for shot in player_data['Shot_Type']]
        ax3.scatter(player_data.index, [player] * len(player_data), 
                   c=shot_colors, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Shot Sequence', fontsize=12)
    ax3.set_ylabel('Player', fontsize=12)
    ax3.set_title('Shot Type Pattern Over Match', fontsize=14, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#FF6B6B', label='Forehand'),
                      Patch(facecolor='#4ECDC4', label='Backhand')]
    ax3.legend(handles=legend_elements, loc='upper right')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Cumulative shot count
    for player in df_sorted['Player'].unique():
        player_data = df_sorted[df_sorted['Player'] == player].reset_index(drop=True)
        cumulative = range(1, len(player_data) + 1)
        ax4.plot(player_data.index, cumulative, linewidth=3, label=player, marker='o', 
                markersize=4, alpha=0.7)
    ax4.set_xlabel('Shot Sequence', fontsize=12)
    ax4.set_ylabel('Cumulative Shot Count', fontsize=12)
    ax4.set_title('Cumulative Shots Over Match', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '07_timeline_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Created: 07_timeline_analysis.png")
    plt.close()


def create_statistics_summary(df, output_dir):
    """Create a visual statistics summary table."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Calculate statistics
    stats_data = []
    
    for player in sorted(df['Player'].unique()):
        player_df = df[df['Player'] == player]
        
        stats_data.append([
            player,
            len(player_df),
            f"{player_df['Max_Racket_Velocity_px_s'].mean():.1f}",
            f"{player_df['Max_Racket_Velocity_px_s'].max():.1f}",
            f"{player_df['Min_Hip_Knee_Angle_deg'].mean():.1f}°",
            f"{player_df['Min_Elbow_Angle_deg'].mean():.1f}°",
            f"{player_df['CoG_Horizontal_Movement_px'].mean():.1f}",
            f"{player_df['Duration_Seconds'].mean():.2f}s",
            f"{len(player_df[player_df['Shot_Type']=='Forehand'])}",
            f"{len(player_df[player_df['Shot_Type']=='Backhand'])}"
        ])
    
    # Add overall row
    stats_data.append([
        'OVERALL',
        len(df),
        f"{df['Max_Racket_Velocity_px_s'].mean():.1f}",
        f"{df['Max_Racket_Velocity_px_s'].max():.1f}",
        f"{df['Min_Hip_Knee_Angle_deg'].mean():.1f}°",
        f"{df['Min_Elbow_Angle_deg'].mean():.1f}°",
        f"{df['CoG_Horizontal_Movement_px'].mean():.1f}",
        f"{df['Duration_Seconds'].mean():.2f}s",
        f"{len(df[df['Shot_Type']=='Forehand'])}",
        f"{len(df[df['Shot_Type']=='Backhand'])}"
    ])
    
    columns = ['Player', 'Total\nShots', 'Avg\nVelocity', 'Max\nVelocity', 
               'Avg Hip/Knee\nAngle', 'Avg Elbow\nAngle', 'Avg CoG\nMovement',
               'Avg\nDuration', 'Forehand\nCount', 'Backhand\nCount']
    
    table = ax.table(cellText=stats_data, colLabels=columns,
                    cellLoc='center', loc='center',
                    colColours=['#4ECDC4']*len(columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#2C3E50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(1, len(stats_data)):
        for j in range(len(columns)):
            if i == len(stats_data) - 1:  # Overall row
                table[(i, j)].set_facecolor('#F39C12')
                table[(i, j)].set_text_props(weight='bold')
            elif i % 2 == 0:
                table[(i, j)].set_facecolor('#ECF0F1')
    
    plt.title('Performance Statistics Summary', fontsize=18, fontweight='bold', pad=20)
    plt.savefig(output_dir / '08_statistics_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Created: 08_statistics_summary.png")
    plt.close()


def main():
    print("\n" + "="*60)
    print("  📊 Datathon Visualization Generator")
    print("="*60)
    
    # Create output directory for visualizations
    output_dir = Path("output/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_latest_csv()
    if df is None:
        return
    
    print(f"\nGenerating visualizations...")
    print("-"*60)
    
    # Generate all visualizations
    create_shot_distribution_pie(df, output_dir)
    create_velocity_analysis(df, output_dir)
    create_angle_analysis(df, output_dir)
    create_shot_duration_analysis(df, output_dir)
    create_player_comparison(df, output_dir)
    create_correlation_heatmap(df, output_dir)
    create_timeline_analysis(df, output_dir)
    create_statistics_summary(df, output_dir)
    
    print("-"*60)
    print(f"\n✅ All visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    for i, name in enumerate([
        '01_shot_distribution.png',
        '02_velocity_analysis.png',
        '03_angle_analysis.png',
        '04_duration_analysis.png',
        '05_player_comparison.png',
        '06_correlation_heatmap.png',
        '07_timeline_analysis.png',
        '08_statistics_summary.png'
    ], 1):
        print(f"  {i}. {name}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
