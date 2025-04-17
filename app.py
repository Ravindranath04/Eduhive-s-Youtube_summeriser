from flask import Flask, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
from groq import Groq
from flask_cors import CORS
import os
import re
from dotenv import load_dotenv
import whisper
import yt_dlp
from tempfile import mkdtemp
import shutil
import cv2
import numpy as np
import base64
from PIL import Image
import io
from datetime import timedelta

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Enable CORS for all routes and all origins for development
CORS(app)

# Initialize clients
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
whisper_model = whisper.load_model("base")

MODEL_LINEUP = ["llama3-70b-8192", "llama3-8b-8192", "gemma-7b-it"]

def extract_video_id(url):
    """Extract video ID from any YouTube URL format"""
    regex = r"(?:v=|\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})"
    match = re.search(regex, url)
    return match.group(1) if match else None

def download_audio(video_url):
    """Robust audio downloader with cleanup"""
    temp_dir = mkdtemp()
    audio_path = None
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
            }],
            'quiet': True,
            'extract_audio': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            audio_path = os.path.join(temp_dir, "audio.mp3")
            return audio_path, info.get('title', '')
    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise Exception(f"Audio download failed: {str(e)}")
    finally:
        # Note: We don't delete the temp_dir here as we need the audio file
        # It will be deleted after transcription in the main function
        pass

def extract_frames_from_video(video_url, max_frames=20):
    """Extract frames from video that might contain diagrams"""
    temp_dir = mkdtemp()
    try:
        # Download video with frame extraction
        video_path = os.path.join(temp_dir, "video.mp4")
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
            'outtmpl': video_path,
            'quiet': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            duration = info.get('duration', 0)  # Video duration in seconds
            
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Failed to open video file")
        
        # Calculate frame sampling interval to get a good distribution
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Sample frames at regular intervals
        frames_to_extract = min(max_frames, total_frames)
        interval = total_frames // frames_to_extract
        
        diagram_frames = []
        
        for i in range(frames_to_extract):
            frame_position = i * interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            # Analyze frame for diagram characteristics
            if is_likely_diagram(frame):
                # Calculate timestamp for this frame
                frame_time = frame_position / fps
                timestamp = str(timedelta(seconds=int(frame_time)))
                
                # Convert frame to base64 string for transmission
                _, buffer = cv2.imencode('.jpg', frame)
                img_str = base64.b64encode(buffer).decode('utf-8')
                
                diagram_frames.append({
                    "image": img_str,
                    "timestamp": timestamp,
                    "confidence": calculate_diagram_confidence(frame)
                })
        
        cap.release()
        return diagram_frames
        
    except Exception as e:
        raise Exception(f"Frame extraction failed: {str(e)}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def is_likely_diagram(frame):
    """Determine if a frame likely contains a diagram or flowchart"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Apply morphological operations to enhance straight lines
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Detect straight lines using Hough transform
    lines = cv2.HoughLinesP(dilated, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    # Calculate percentage of edge pixels
    edge_percentage = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Criteria for a diagram:
    # 1. Has enough straight lines (diagrams often have more straight lines)
    # 2. Has moderate edge percentage (not too cluttered, not too empty)
    # 3. Has limited color variation (diagrams often use a limited color palette)
    
    if lines is not None and len(lines) > 15:
        # Check color variance in original image
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_var = np.var(hsv[:,:,0])
        s_var = np.var(hsv[:,:,1])
        
        # Diagrams typically have low to moderate hue and saturation variance
        # Combined with enough straight lines and proper edge percentage
        if 0.01 < edge_percentage < 0.2 and h_var < 2000 and s_var < 5000:
            return True
    
    return False

def calculate_diagram_confidence(frame):
    """Calculate confidence that the frame contains a diagram (0-100%)"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Detect straight lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    # Calculate edge percentage
    edge_percentage = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # More sophisticated checks for diagram-like characteristics
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_var = np.var(hsv[:,:,0])
    s_var = np.var(hsv[:,:,1])
    v_var = np.var(hsv[:,:,2])
    
    # Weights for different factors
    line_weight = 0.4
    edge_weight = 0.3
    color_weight = 0.3
    
    # Calculate individual scores
    line_score = min(100, (lines.shape[0] / 30) * 100) if lines is not None else 0
    edge_score = 100 if 0.01 < edge_percentage < 0.15 else max(0, 100 - abs(edge_percentage - 0.08) * 1000)
    color_score = 100 - min(100, (h_var / 2000) * 100)
    
    # Weighted average
    confidence = (line_weight * line_score + edge_weight * edge_score + color_weight * color_score)
    
    return min(100, max(0, int(confidence)))

def analyze_diagrams_with_ai(diagrams, video_title):
    """Use Groq to analyze and describe the detected diagrams"""
    if not diagrams:
        return []
    
    descriptions = []
    
    # Sample prompt for diagram analysis
    system_prompt = """
    You are an expert at analyzing diagrams, flowcharts, and visual content from educational videos.
    For each diagram image, provide:
    1. A concise description of what the diagram represents
    2. The key concepts or relationships shown
    3. How this diagram fits into the overall video topic
    Keep your analysis under 100 words per diagram.
    """
    
    for i, diagram in enumerate(diagrams):
        if diagram["confidence"] < 40:  # Skip low confidence diagrams
            continue
            
        try:
            for model in MODEL_LINEUP:
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": f"This is a diagram from the video titled '{video_title}'. Based on the image and timestamp ({diagram['timestamp']}), describe what this diagram represents and how it fits into the video topic. Note that you don't actually see the image - please provide a general analysis based on the video title and timestamp."
                            }
                        ],
                        temperature=0.4,
                        max_tokens=200
                    )
                    
                    descriptions.append({
                        "timestamp": diagram["timestamp"],
                        "confidence": diagram["confidence"],
                        "image": diagram["image"],
                        "description": response.choices[0].message.content
                    })
                    break
                except Exception:
                    continue
        except Exception as e:
            print(f"Error analyzing diagram {i}: {str(e)}")
    
    return descriptions

def generate_flowchart(summary):
    """Generate enhanced flowchart with improved design elements and visual appeal"""
    
    flowchart_template = """
    Create a visually stunning ASCII art flowchart that captures the key concepts from the summary.
    
    Requirements:
    1. Use rich box drawing characters (─, │, ┌, ┐, └, ┘, ├, ┤, ┬, ┴, ┼) to create professional-looking boxes and connections
    2. Implement a clear hierarchical structure showing relationships between concepts
    3. Use decorative elements to enhance visual appeal:
       - Use arrows (→, ↓, ↑, ←, ↗, ↘, ↙, ↖) to show directional relationships
       - Use stars (★) to highlight the most important concepts
       - Use other symbols like ○, ●, ◆, □, ■ for different types of nodes
       - Add double lines (═, ║, ╔, ╗, ╚, ╝) for main concept boxes
    4. Create a balanced layout with proper spacing and alignment
    5. Include a title box at the top with a double-lined border
    6. Use indentation consistently to show hierarchical relationships
    7. Highlight [MC] marked concepts with special formatting
    
    Layout Example:
    
    ╔═════════════════════════════════════╗
    ║           CONCEPT FLOWMAP           ║
    ╚═════════════════════════════════════╝
                      │
                      ↓
         ┌────────────┴─────────────┐
         │                          │
         ↓                          ↓
    ┌─────────┐               ┌─────────┐
    │ ★ KEY   │               │ MAJOR   │
    │ CONCEPT │──────────────→│ CONCEPT │
    └────┬────┘               └────┬────┘
         │                         │
         ↓                         ↓
    ┌─────────┐               ┌─────────┐
    │ ○ Sub   │               │ ○ Sub   │
    │ Point 1 │               │ Point 2 │
    └─────────┘               └─────────┘
    
    Create a professional-looking flowchart that effectively visualizes the key relationships between concepts in the summary.
    Focus on making the diagram clear, well-structured, and visually appealing.
    """
    
    for model in MODEL_LINEUP:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": flowchart_template
                    },
                    {
                        "role": "user",
                        "content": f"Create a visually appealing flowchart for this summary:\n\n{summary}"
                    }
                ],
                temperature=0.3,
                max_tokens=1500  # Increased token limit for more detailed flowcharts
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error with model {model}: {str(e)}")
            continue
    
    # If all models fail, return a simple error message
    return "Flowchart generation failed. Please try again later."

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        # Validate input
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({"error": "YouTube URL required"}), 400
        
        video_url = data['url']
        video_id = extract_video_id(video_url)
        if not video_id:
            return jsonify({"error": "Invalid YouTube URL"}), 400

        # Get configuration parameters with validation
        num_points = min(max(int(data.get('points', 5)), 1), 10)
        detect_diagrams = data.get('detect_diagrams', True)  # New option, defaults to True

        # Content extraction pipeline
        transcript_text = ""
        method_used = "subtitles"
        video_title = "Unknown Title"
        audio_path = None
        temp_dir = None
        
        try:
            # Try official subtitles first
            transcript = YouTubeTranscriptApi.get_transcript(
                video_id,
                languages=['en', '*']  # English and fallback
            )
            transcript_text = " ".join([line['text'] for line in transcript])
        except Exception as subtitle_error:
            print(f"Subtitle extraction failed: {str(subtitle_error)}")
            # Fallback to audio transcription
            try:
                audio_path, video_title = download_audio(video_url)
                temp_dir = os.path.dirname(audio_path)
                result = whisper_model.transcribe(audio_path)
                transcript_text = result["text"]
                method_used = "audio_transcription"
            except Exception as audio_error:
                return jsonify({
                    "error": "Content extraction failed",
                    "details": str(audio_error),
                    "solutions": [
                        "Try video with enabled subtitles",
                        "Check video availability",
                        "Try shorter video (<15 mins)"
                    ]
                }), 400
            finally:
                # Clean up audio files if they exist
                if audio_path and os.path.exists(audio_path):
                    try:
                        os.remove(audio_path)
                    except:
                        pass
                if temp_dir and os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir)
                    except:
                        pass

        # Generate summary with model fallback
        summary = ""
        used_model = "unknown"
        
        # System prompt
        system_prompt = f"Generate {num_points} key points:\n- Prioritize technical content\n- Include statistics and numbers\n- Mark main concepts with [MC]\n- Use concise bullet points"
        
        for model in MODEL_LINEUP:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": transcript_text[:15000]  # First 15k chars to avoid token limits
                        }
                    ],
                    temperature=0.4,
                    max_tokens=800
                )
                summary = response.choices[0].message.content
                used_model = model
                break
            except Exception as e:
                print(f"Model {model} failed: {str(e)}")
                continue

        if not summary:
            return jsonify({"error": "All AI models failed"}), 500

        # Generate enhanced flowchart
        flowchart = generate_flowchart(summary)

        # If we used audio transcription but didn't get the title from yt-dlp
        if video_title == "Unknown Title":
            try:
                # Try to get the title from YouTube API
                with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                    info = ydl.extract_info(video_url, download=False)
                    video_title = info.get('title', 'Unknown Title')
            except:
                pass
        
        # Extract and analyze diagrams from video frames if feature is enabled
        diagrams = []
        if detect_diagrams:
            try:
                print("Extracting diagrams from video frames...")
                extracted_diagrams = extract_frames_from_video(video_url)
                if extracted_diagrams:
                    diagrams = analyze_diagrams_with_ai(extracted_diagrams, video_title)
                    print(f"Found {len(diagrams)} potential diagrams/flowcharts")
            except Exception as e:
                print(f"Diagram extraction failed: {str(e)}")
                # Don't fail the entire request if just the diagram extraction fails
                pass

        return jsonify({
            "success": True,
            "summary": summary,
            "flowchart": flowchart,
            "video_id": video_id,
            "video_title": video_title,
            "method": method_used,
            "model": used_model,
            "points_generated": num_points,
            "diagrams": diagrams  # New field containing extracted diagrams
        })

    except Exception as e:
        print(f"Error in summarize route: {str(e)}")
        return jsonify({
            "error": "Processing failed",
            "details": str(e)
        }), 500

@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "endpoints": {
            "POST /summarize": "Generate summary with flowchart and diagrams",
            "parameters": {
                "url": "YouTube URL (required)",
                "points": "Number of key points (1-10, default 5)",
                "detect_diagrams": "Extract diagrams from video (boolean, default true)"
            }
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)