from flask import Flask, request, jsonify, send_file, send_from_directory, Response
from flask_cors import CORS
import os
import csv
import subprocess
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(BASE_DIR, 'input')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'output')
OUTPUT_VIDEO_FOLDER = os.path.join(OUTPUT_FOLDER, 'analysisVideo')
DIAGRAMS_FOLDER = os.path.join(OUTPUT_FOLDER, 'diagrams')
CSV_FOLDER = os.path.join(OUTPUT_FOLDER, 'analysisCSV')

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ensure_folders():
    """Ensure all necessary folders exist"""
    for folder in [INPUT_FOLDER, OUTPUT_VIDEO_FOLDER, DIAGRAMS_FOLDER, CSV_FOLDER]:
        os.makedirs(folder, exist_ok=True)

@app.route('/')
def index():
    return jsonify({
        'message': 'Rallymate Backend API',
        'version': '1.0',
        'endpoints': {
            'POST /analyze': 'Upload and analyze video',
            'GET /videos': 'List output videos',
            'GET /video/<filename>': 'Stream video file',
            'GET /diagrams': 'List diagram files',
            'GET /diagram/<filename>': 'Get diagram image',
            'GET /csv-files': 'List CSV files',
            'GET /csv/<filename>': 'Get CSV data as JSON'
        }
    })

@app.route('/analyze', methods=['POST'])
def analyze_video():
    """Handle video upload and trigger analysis"""
    ensure_folders()
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed types: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        input_filename = f'input_{timestamp}_{filename}'
        input_path = os.path.join(INPUT_FOLDER, input_filename)
        file.save(input_path)
        
        # Run analysis script
        # Assuming you have a main analysis script like analyze_processed_video.py
        analysis_script = os.path.join(BASE_DIR, 'analyze_processed_video.py')
        
        if os.path.exists(analysis_script):
            # Run the analysis script with the input video
            result = subprocess.run(
                ['python', analysis_script, input_path],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode != 0:
                return jsonify({
                    'error': 'Analysis failed',
                    'details': result.stderr
                }), 500
        
        return jsonify({
            'success': True,
            'message': 'Video analyzed successfully',
            'input_file': input_filename
        }), 200
        
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Analysis timeout - video may be too long'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/videos', methods=['GET'])
def list_videos():
    """List all output videos"""
    ensure_folders()
    
    try:
        videos = []
        if os.path.exists(OUTPUT_VIDEO_FOLDER):
            videos = [f for f in os.listdir(OUTPUT_VIDEO_FOLDER) 
                     if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            videos.sort(reverse=True)  # Most recent first
        
        return jsonify(videos), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/video/<filename>', methods=['GET'])
def get_video(filename):
    """Stream video file with range support"""
    try:
        safe_filename = secure_filename(filename)
        video_path = os.path.join(OUTPUT_VIDEO_FOLDER, safe_filename)
        
        print(f"Video request for: {filename}")
        print(f"Safe filename: {safe_filename}")
        print(f"Full path: {video_path}")
        print(f"File exists: {os.path.exists(video_path)}")
        
        if not os.path.exists(video_path):
            return jsonify({'error': 'Video not found', 'path': video_path}), 404
        
        # Get file size
        file_size = os.path.getsize(video_path)
        
        # Check if range request
        range_header = request.headers.get('Range', None)
        
        if not range_header:
            # No range request, send entire file
            return send_file(
                video_path,
                mimetype='video/mp4',
                as_attachment=False,
                download_name=safe_filename
            )
        
        # Parse range header
        byte_start, byte_end = 0, file_size - 1
        match = range_header.replace('bytes=', '').split('-')
        if match[0]:
            byte_start = int(match[0])
        if match[1]:
            byte_end = int(match[1])
        
        length = byte_end - byte_start + 1
        
        # Read the requested chunk
        with open(video_path, 'rb') as f:
            f.seek(byte_start)
            data = f.read(length)
        
        response = Response(
            data,
            206,  # Partial Content
            mimetype='video/mp4',
            direct_passthrough=True
        )
        
        response.headers.add('Content-Range', f'bytes {byte_start}-{byte_end}/{file_size}')
        response.headers.add('Accept-Ranges', 'bytes')
        response.headers.add('Content-Length', str(length))
        
        return response
        
    except Exception as e:
        print(f"Error serving video: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/diagrams', methods=['GET'])
def list_diagrams():
    """List all diagram files"""
    ensure_folders()
    
    try:
        diagrams = []
        if os.path.exists(DIAGRAMS_FOLDER):
            diagrams = [f for f in os.listdir(DIAGRAMS_FOLDER) 
                       if f.endswith(('.png', '.jpg', '.jpeg', '.svg'))]
            diagrams.sort(reverse=True)
        
        return jsonify(diagrams), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/diagram/<filename>', methods=['GET'])
def get_diagram(filename):
    """Get diagram image"""
    try:
        diagram_path = os.path.join(DIAGRAMS_FOLDER, secure_filename(filename))
        if not os.path.exists(diagram_path):
            return jsonify({'error': 'Diagram not found'}), 404
        
        return send_file(diagram_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/csv-files', methods=['GET'])
def list_csv_files():
    """List all CSV files"""
    ensure_folders()
    
    try:
        csv_files = []
        if os.path.exists(CSV_FOLDER):
            csv_files = [f for f in os.listdir(CSV_FOLDER) if f.endswith('.csv')]
            csv_files.sort(reverse=True)
        
        return jsonify(csv_files), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/csv/<filename>', methods=['GET'])
def get_csv_data(filename):
    """Get CSV data as JSON"""
    try:
        csv_path = os.path.join(CSV_FOLDER, secure_filename(filename))
        if not os.path.exists(csv_path):
            return jsonify({'error': 'CSV file not found'}), 404
        
        data = []
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
        
        return jsonify(data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    ensure_folders()
    print('Starting Rallymate Backend Server...')
    print(f'Base Directory: {BASE_DIR}')
    print(f'Input Folder: {INPUT_FOLDER}')
    print(f'Output Folder: {OUTPUT_FOLDER}')
    app.run(debug=True, host='0.0.0.0', port=5000)
