# Rallymate
Automatically analyze your table tennis games and provide feedback

## 🎯 Project Overview
Rallymate is a comprehensive table tennis analysis system that uses computer vision and pose estimation to provide detailed biomechanical insights into gameplay. Built for datathon presentation, it processes video footage to generate analytical reports, annotated demonstration videos, and rich data visualizations.

## ✨ Features
- **Video Analysis**: Processes table tennis match videos using YOLOv8 pose estimation
- **Shot Detection**: Automatically detects and classifies Forehand and Backhand shots
- **Biomechanical Metrics**: Analyzes:
  - Racket velocity (max and average)
  - Joint angles (hip/knee, elbow, torso)
  - Center of gravity movement
  - Shot duration and timing
- **Detailed Reports**: Generates TXT and CSV reports with comprehensive statistics
- **Annotated Videos**: Creates demonstration videos with:
  - 17 COCO keypoint visualization
  - Full pose skeleton overlay
  - Real-time biomechanical tracking
- **Rich Visualizations**: Generates 8+ matplotlib/seaborn diagrams including:
  - Shot distribution analysis
  - Velocity analysis (distribution, trends, comparisons)
  - Joint angle analysis
  - Duration analysis
  - Player-to-player comparisons
  - Correlation heatmaps
  - Timeline analysis
  - Statistics summary tables

## 📂 Project Structure
```
Rallymate/
├── analyze_processed_video.py    # Main analysis script (generates TXT/CSV)
├── render_analysis_video.py      # Video rendering with pose visualization
├── visualize_analysis.py         # Generates datathon-ready diagrams
├── requirements.txt              # Python dependencies
├── input/                        # Input videos
│   └── dataDetection.mp4
├── output/
│   ├── analysisCSV/             # CSV analysis reports
│   ├── analysisText/            # Text analysis reports
│   ├── analysisVideo/           # Annotated demonstration videos
│   └── visualizations/          # Generated diagrams (8 PNG files)
└── src/
    └── vision/                   # Core analysis modules
        ├── game_analyzer.py
        ├── pose_analyzer.py
        └── ...
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Git LFS (for large video files)

### Installation
```bash
# Clone the repository
git clone https://github.com/Ashwani564/DataThonProject.git
cd DataThonProject

# Install dependencies
pip install -r requirements.txt

# Initialize Git LFS (if not already done)
git lfs install
```

### Usage

#### 1. Analyze a Video (Generate Reports)
```bash
python analyze_processed_video.py
```
This generates:
- `output/analysisText/analysis_[video]_[timestamp].txt`
- `output/analysisCSV/analysis_[video]_[timestamp].csv`

#### 2. Render Annotated Video
```bash
python render_analysis_video.py
```
This generates:
- `output/analysisVideo/analyzed_video.mp4` (with pose skeletons and keypoints)

#### 3. Generate Visualizations
```bash
python visualize_analysis.py
```
This generates 8 diagrams in `output/visualizations/`:
1. `01_shot_distribution.png` - Shot type distribution
2. `02_velocity_analysis.png` - Velocity metrics
3. `03_angle_analysis.png` - Joint angle analysis
4. `04_duration_analysis.png` - Shot duration analysis
5. `05_player_comparison.png` - Player performance comparison
6. `06_correlation_heatmap.png` - Metric correlations
7. `07_timeline_analysis.png` - Timeline and trends
8. `08_statistics_summary.png` - Summary table

## 📊 Sample Output

### Analysis CSV Format
```csv
Player,Shot_Number,Shot_Type,Start_Frame,End_Frame,Duration_Seconds,Max_Racket_Velocity_px_s,Avg_Racket_Velocity_px_s,Min_Hip_Knee_Angle_deg,Avg_Hip_Knee_Angle_deg,Min_Elbow_Angle_deg,Avg_Elbow_Angle_deg,Avg_Torso_Angle_deg,CoG_Horizontal_Movement_px,CoG_Vertical_Movement_px
Player_1,1,Forehand,13,14,0.03,54.3,27.15,31.54,31.67,89.35,92.13,8.76,30.66,1.19
Player_1,2,Backhand,27,31,0.13,1281.12,779.44,5.13,21.24,52.07,79.84,29.18,31.66,15.98
...
```

## 🎨 Visualization Examples
All diagrams use a consistent color scheme:
- Forehand shots: Red (#FF6B6B)
- Backhand shots: Teal (#4ECDC4)
- Professional styling with grid overlays and clear legends

## 🛠️ Technologies Used
- **YOLOv8**: Pose estimation and keypoint detection
- **OpenCV**: Video processing and rendering
- **Pandas**: Data analysis and manipulation
- **Matplotlib & Seaborn**: Data visualization
- **NumPy**: Numerical computations

## 📝 Key Metrics Analyzed
- **Velocity**: Max and average racket velocity per shot
- **Joint Angles**: Hip/knee, elbow, and torso angles
- **Movement**: Horizontal and vertical center of gravity displacement
- **Timing**: Shot duration and frame-level timing
- **Shot Classification**: Automatic forehand/backhand detection

## 🎓 Datathon Ready
This project is optimized for datathon presentations with:
- Comprehensive documentation
- Rich visualizations
- Clean, reproducible code
- Sample data and outputs included
- Git LFS for large file management

## 📄 License
This project is available for educational and datathon purposes.

## 👥 Contributors
- Ashwani Kumar ([@Ashwani564](https://github.com/Ashwani564))

## 🔗 Repository
https://github.com/Ashwani564/DataThonProject

---
**Note**: All video files are tracked using Git LFS to handle large file sizes efficiently.
