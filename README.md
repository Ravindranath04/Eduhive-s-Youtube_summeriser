# EDUHIVE'S YouTube Video Summarizer

A powerful tool that transforms any YouTube video into concise summaries, visual flowcharts, and extracts diagrams directly from the video content. Perfect for quick learning, note-taking, and content analysis.

![YouTube Summarizer Preview](https://via.placeholder.com/800x400)

## Features

- **Key Point Summary**: Extract the most important information from any YouTube video
- **Visual Flowchart**: Convert video content into an intuitive flowchart for better understanding
- **Diagram Extraction**: Automatically identify and extract diagrams, charts, and visual content directly from the video
- **AI-Powered Analysis**: Intelligent processing using Groq API and computer vision

## Demo

![Demo GIF](https://via.placeholder.com/600x400)

## Installation

### Prerequisites

- Python 3.8+
- Node.js (for front-end development)
- FFmpeg (required for video processing)

### Installing FFmpeg

FFmpeg is essential for the diagram extraction feature. Install it based on your operating system:

#### Windows
1. Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract the files to a directory (e.g., `C:\ffmpeg`)
3. Add FFmpeg to your system PATH:
   - Right-click on "This PC" → Properties → Advanced system settings → Environment Variables
   - Edit the PATH variable and add the path to the FFmpeg `bin` folder (e.g., `C:\ffmpeg\bin`)
   - Click OK to save changes

#### macOS
```bash
brew install ffmpeg
```

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install ffmpeg
```

#### CentOS/RHEL
```bash
sudo yum install epel-release
sudo yum install ffmpeg ffmpeg-devel
```

Verify installation by running:
```bash
ffmpeg -version
```

### Setup Backend

1. Clone the repository:
```bash
git clone https://github.com/yourusername/youtube-summarizer.git
cd youtube-summarizer
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install Python dependencies:
```bash
pip install flask flask-cors youtube-transcript-api groq python-dotenv whisper yt-dlp opencv-python numpy pillow
```

4. Create a `.env` file in the project root with your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

5. Start the Flask server:
```bash
python app.py
```

The API will be available at `http://localhost:5000`.

### Setup Frontend

1. Open another terminal and navigate to the frontend directory
2. Open the HTML file in a web browser or use a simple HTTP server:
```bash
# Using Python's built-in HTTP server
python -m http.server 8000
```

3. Access the application at `http://localhost:8000`

## Usage

1. Enter a YouTube URL in the input field
2. Select the number of summary points you want (3, 5, 7, or 10)
3. Check/uncheck the "Extract diagrams from video" option
4. Click "Generate All"
5. View the results in the three tabs:
   - Summary: Key points extracted from the video
   - Flowchart: Visual representation of concepts
   - Diagrams: Visual content extracted from the video (requires FFmpeg)

## API Reference

### Endpoints

#### POST /summarize
Generates a summary, flowchart, and extracts diagrams from a YouTube video.

**Request Body:**
```json
{
  "url": "https://www.youtube.com/watch?v=example",
  "points": 5,
  "detect_diagrams": true
}
```

**Response:**
```json
{
  "success": true,
  "summary": "...",
  "flowchart": "...",
  "video_id": "example",
  "video_title": "Example Video",
  "method": "subtitles",
  "model": "llama3-70b-8192",
  "points_generated": 5,
  "diagrams": [...]
}
```

## Troubleshooting

### Common Issues

- **No diagrams detected**: Make sure FFmpeg is properly installed and in your PATH
- **Connection errors**: Verify that the Flask backend is running on port 5000
- **Missing subtitles**: Some videos don't have subtitles; the app will fall back to audio transcription

### Debug Tips

- Check console output for errors
- Ensure your Groq API key is valid
- Try videos with clear visual elements for better diagram extraction
- For video processing issues, verify FFmpeg installation

## Technical Details

The application uses several key technologies:

- **Flask**: Backend API server
- **Groq API**: AI text generation for summaries and flowcharts
- **YouTube Transcript API**: Extract subtitles from videos
- **OpenCV**: Computer vision for diagram detection
- **Whisper**: Audio transcription when subtitles aren't available
- **yt-dlp**: YouTube video downloading

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The AI models provided by Groq
- YouTube Transcript API for subtitle extraction
- OpenAI's Whisper for audio transcription
- OpenCV for computer vision capabilities
