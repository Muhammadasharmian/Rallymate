"""
Run game analysis on processed video output.
This script analyzes the video and generates detailed TXT and CSV reports.
For video rendering with pose visualization, use render_analysis_video.py
"""
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from vision.game_analyzer import GameAnalyzer


def main():
    
    # Ensure output directories exist
    Path("output/analysisCSV").mkdir(parents=True, exist_ok=True)
    Path("output/analysisText").mkdir(parents=True, exist_ok=True)
    Path("output/analysisVideo").mkdir(parents=True, exist_ok=True)
    
    # Look for video in input/ directory
    input_dir = Path("input")
    if not input_dir.exists():
        print(f"Error: Directory 'input/' not found!")
        return
    
    mp4_files = list(input_dir.glob("*.mp4"))
    if not mp4_files:
        print(f"Error: No .mp4 files found in 'input/' directory!")
        return
    
    # Use the first mp4 file found
    video_path = str(mp4_files[0])
    print(f"Found video: {video_path}")
    
    # Check if video exists
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        print("\nPlease check the file path and try again.")
        print("Or specify a different video path as argument:")
        print(f"  python analyze_processed_video.py <video_path>")
        return
    
    # FPS and velocity threshold (can be adjusted based on video)
    fps = 30  # Adjust if your video has different FPS
    velocity_threshold = 1000.0  # Minimum velocity to detect a shot
    
    print("\n" + "="*60)
    print("  🏓 PaddleCoach - Game Analysis System")
    print("="*60)
    print(f"\nInput Video: {video_path}")
    print(f"Processing FPS: {fps}")
    print(f"Shot Detection Threshold: {velocity_threshold} px/s")
    print("\nThis analysis will:")
    print("  ✅ Extract pose data from the video")
    print("  ✅ Detect shots for both players")
    print("  ✅ Calculate biomechanical metrics")
    print("  ✅ Compare player performance")
    print("  ✅ Generate TXT and CSV reports")
    print("-"*60)
    
    # Create analyzer with correct output directories
    analyzer = GameAnalyzer(output_dir="output/analysisText")
    
    # Generate analysis reports (TXT and CSV)
    print("\nPhase 1: Analyzing video and generating reports...")
    result = analyzer.generate_analysis_report(
        video_path=video_path,
        velocity_threshold=velocity_threshold,
        fps=fps
    )
    
    if result:
        report_path, csv_path = result
        print("\n" + "="*60)
        print("  ✅ ANALYSIS COMPLETE!")
        print("="*60)
        print(f"\n📄 TXT Report saved to: {report_path}")
        print(f"📊 CSV Report saved to: {csv_path}")
        print("\nThe reports include:")
        print("  • Shot detection summary")
        print("  • Forehand/Backhand classification")
        print("  • Biomechanical metrics per shot")
        print("  • Player comparison (if 2 players detected)")
        print("  • Performance recommendations (TXT)")
        print("  • Complete shot data in CSV format (all shots)")
        print("\n" + "="*60)
        print("\n💡 To generate an annotated video with pose visualization:")
        print("   Run: python3 render_analysis_video.py")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("  ❌ ANALYSIS FAILED")
        print("="*60)
        print("\nPlease check:")
        print("  • Video file exists and is readable")
        print("  • Video contains pose detection data")
        print("  • YOLO model (yolo11n-pose.pt) is available")
        print("="*60)


if __name__ == "__main__":
    main()
