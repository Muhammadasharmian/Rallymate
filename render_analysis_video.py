"""
Render annotated video with pose detection and analysis overlay.
This script generates a demonstration video showing pose estimation with skeletal lines
and real-time analysis data.
"""
from pathlib import Path
import sys
import cv2
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from vision.game_analyzer import GameAnalyzer
from ultralytics import YOLO


# COCO 17 Keypoint indices
KP = {
    'NOSE': 0, 'LEFT_EYE': 1, 'RIGHT_EYE': 2, 'LEFT_EAR': 3, 'RIGHT_EAR': 4,
    'LEFT_SHOULDER': 5, 'RIGHT_SHOULDER': 6,
    'LEFT_ELBOW': 7, 'RIGHT_ELBOW': 8, 'LEFT_WRIST': 9, 'RIGHT_WRIST': 10,
    'LEFT_HIP': 11, 'RIGHT_HIP': 12,
    'LEFT_KNEE': 13, 'RIGHT_KNEE': 14, 'LEFT_ANKLE': 15, 'RIGHT_ANKLE': 16,
}

# Define skeleton connections (bone pairs)
SKELETON_CONNECTIONS = [
    # Head
    (KP['NOSE'], KP['LEFT_EYE']),
    (KP['NOSE'], KP['RIGHT_EYE']),
    (KP['LEFT_EYE'], KP['LEFT_EAR']),
    (KP['RIGHT_EYE'], KP['RIGHT_EAR']),
    # Torso
    (KP['LEFT_SHOULDER'], KP['RIGHT_SHOULDER']),
    (KP['LEFT_SHOULDER'], KP['LEFT_HIP']),
    (KP['RIGHT_SHOULDER'], KP['RIGHT_HIP']),
    (KP['LEFT_HIP'], KP['RIGHT_HIP']),
    # Arms
    (KP['LEFT_SHOULDER'], KP['LEFT_ELBOW']),
    (KP['LEFT_ELBOW'], KP['LEFT_WRIST']),
    (KP['RIGHT_SHOULDER'], KP['RIGHT_ELBOW']),
    (KP['RIGHT_ELBOW'], KP['RIGHT_WRIST']),
    # Legs
    (KP['LEFT_HIP'], KP['LEFT_KNEE']),
    (KP['LEFT_KNEE'], KP['LEFT_ANKLE']),
    (KP['RIGHT_HIP'], KP['RIGHT_KNEE']),
    (KP['RIGHT_KNEE'], KP['RIGHT_ANKLE']),
]


def draw_pose_on_frame(frame, keypoints, confidence_threshold=0.5):
    """
    Draw pose skeleton and keypoints on frame.
    
    Args:
        frame: Image frame
        keypoints: Array of keypoints [17, 3] (x, y, confidence)
        confidence_threshold: Minimum confidence to draw keypoint
    """
    # Draw skeleton lines
    for start_idx, end_idx in SKELETON_CONNECTIONS:
        if (keypoints[start_idx, 2] > confidence_threshold and 
            keypoints[end_idx, 2] > confidence_threshold):
            
            start_point = (int(keypoints[start_idx, 0]), int(keypoints[start_idx, 1]))
            end_point = (int(keypoints[end_idx, 0]), int(keypoints[end_idx, 1]))
            
            # Draw line (bone)
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
    
    # Draw keypoints (joints)
    for i in range(17):
        if keypoints[i, 2] > confidence_threshold:
            x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
            # Draw filled circle for joint
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            # Draw outer circle
            cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)


def generate_video_with_pose(video_path: str, analyzer: GameAnalyzer,
                             velocity_threshold: float, fps: int,
                             output_path: str = "output/analysisVideo/analyzed_video.mp4"):
    """
    Generate annotated video with pose visualization and analysis overlay.
    
    Args:
        video_path: Path to input video
        analyzer: GameAnalyzer instance
        velocity_threshold: Shot detection threshold
        fps: Frames per second
        output_path: Path to save output video
    """
    print("\n" + "="*60)
    print("  🎥 Generating Annotated Video with Pose Visualization")
    print("="*60)
    print(f"\nInput: {video_path}")
    print(f"Output: {output_path}")
    
    # Load YOLO model for pose detection
    print("\nLoading YOLO pose model...")
    model = YOLO("yolo11n-pose.pt")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return False
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Set up video writer (same size as input)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (frame_width, frame_height))
    
    frame_num = 0
    
    print(f"\nProcessing {total_frames} frames with pose visualization...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        
        # Show progress
        if frame_num % 500 == 0:
            progress = (frame_num / total_frames) * 100
            print(f"  Progress: {progress:.1f}% ({frame_num}/{total_frames} frames)")
        
        # Run pose detection on frame
        results = model(frame, verbose=False)
        
        # Draw poses on frame
        if results[0].keypoints is not None and len(results[0].keypoints) > 0:
            for keypoints in results[0].keypoints.data:
                # keypoints shape: [17, 3] (x, y, confidence)
                draw_pose_on_frame(frame, keypoints.cpu().numpy())
        
        # Write the frame with pose overlay
        out.write(frame)
    
    cap.release()
    out.release()
    
    print("\n" + "="*60)
    print(f"  ✅ Video saved to: {output_path}")
    print("="*60)
    return True


def main():
    # Ensure output directories exist
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
        return
    
    # FPS and velocity threshold
    fps = 30
    velocity_threshold = 1000.0
    
    print("\n" + "="*60)
    print("  🎥 Video Renderer with Pose Visualization")
    print("="*60)
    print(f"\nInput Video: {video_path}")
    print(f"Processing FPS: {fps}")
    print(f"Shot Detection Threshold: {velocity_threshold} px/s")
    
    # Create analyzer
    analyzer = GameAnalyzer(output_dir="output/analysisText")
    
    # Generate video with pose visualization
    success = generate_video_with_pose(
        video_path=video_path,
        analyzer=analyzer,
        velocity_threshold=velocity_threshold,
        fps=fps,
        output_path="output/analysisVideo/analyzed_video.mp4"
    )
    
    if success:
        print("\n✅ Video rendering complete!")
        print("📹 Annotated video with pose visualization saved!")
    else:
        print("\n❌ Video rendering failed!")


if __name__ == "__main__":
    main()
