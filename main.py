from flask import Flask, request, jsonify, send_file
import asyncio
import subprocess
import os
import tempfile
from datetime import datetime
import openai
from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer  # Added missing import
import threading
from functools import wraps
import serial
import time

app = Flask(__name__)
openai_client = AsyncOpenAI()

# 3D Printer connection
printer_port = '/dev/ttyUSB0'  # Adjust if different
printer_baudrate = 115200
printer = None

def get_printer_connection():
    """Get or create printer serial connection"""
    global printer
    if printer is None or not printer.is_open:
        try:
            printer = serial.Serial(printer_port, printer_baudrate, timeout=5)
            time.sleep(2)  # Wait for connection to stabilize
        except Exception as e:
            print(f"Failed to connect to printer: {e}")
            return None
    return printer

def send_gcode(command):
    """Send G-code command to printer and return response"""
    try:
        conn = get_printer_connection()
        if not conn:
            return {"error": "No printer connection"}
        
        # Clear any pending input buffer first
        conn.reset_input_buffer()
        
        # Send command
        conn.write((command + '\n').encode())
        conn.flush()  # Ensure command is sent immediately
        
        # Read response
        response = ""
        start_time = time.time()
        while time.time() - start_time < 10:  # Increased timeout to 10 seconds
            if conn.in_waiting:
                line = conn.readline().decode().strip()
                if line:  # Only add non-empty lines
                    response += line + "\n"
                    if line.startswith('ok') or line.startswith('error'):
                        break
            time.sleep(0.1)  # Small delay to prevent busy waiting
        
        return {"success": True, "command": command, "response": response.strip()}
        
    except Exception as e:
        return {"error": str(e)}

def async_route(f):
    """Decorator to handle async functions in Flask routes"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(f(*args, **kwargs))
        finally:
            loop.close()
    return wrapper

@app.route('/take_picture', methods=['POST'])
def take_picture():
    """
    Endpoint to take a picture using the Pi camera
    Returns: JSON with success status and image file path
    """
    try:
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"photo_{timestamp}.jpg"
        filepath = os.path.join("/tmp", filename)
        
        # Take picture using rpicam-still
        result = subprocess.run([
            "rpicam-still", 
            "-o", filepath,
            "--timeout", "2000",  # 2 second timeout
            "--width", "1920",
            "--height", "1080",
            "-n" # no preview
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            return jsonify({
                "success": True,
                "message": "Picture taken successfully",
                "filename": filename,
                "filepath": filepath,
                "download_url": f"/download_image/{filename}"
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": f"Camera error: {result.stderr}"
            }), 500
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/download_image/<filename>', methods=['GET'])
def download_image(filename):
    """
    Endpoint to download the captured image
    """
    try:
        filepath = os.path.join("/tmp", filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True, download_name=filename)
        else:
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/printer/home', methods=['POST'])
def home_printer():
    """Home all axes"""
    result = send_gcode("G28")
    return jsonify(result)

@app.route('/printer/move', methods=['POST'])
def move_printer():
    """
    Move printer to specified coordinates
    JSON payload: {"x": 100, "y": 50, "z": 10, "speed": 3000}
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Build G1 command
        command_parts = ["G1"]
        
        if 'x' in data:
            command_parts.append(f"X{data['x']}")
        if 'y' in data:
            command_parts.append(f"Y{data['y']}")
        if 'z' in data:
            command_parts.append(f"Z{data['z']}")
        if 'speed' in data:
            command_parts.append(f"F{data['speed']}")
        
        if len(command_parts) == 1:
            return jsonify({"error": "No movement parameters provided"}), 400
        
        command = " ".join(command_parts)
        result = send_gcode(command)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/printer/position', methods=['GET'])
def get_position():
    """Get current printer position"""
    result = send_gcode("M114")  # M114 is correct for position
    return jsonify(result)

@app.route('/printer/status', methods=['GET'])
def printer_status():
    """Get printer status and info"""
    result = send_gcode("M115")  # M115 is correct for firmware info
    return jsonify(result)

@app.route('/printer/command', methods=['POST'])
def send_raw_gcode():
    """
    Send raw G-code command
    JSON payload: {"command": "G28"}
    """
    try:
        data = request.get_json()
        if not data or 'command' not in data:
            return jsonify({"error": "Missing 'command' field"}), 400
        
        result = send_gcode(data['command'])
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/printer/emergency_stop', methods=['POST'])
def emergency_stop():
    """Emergency stop - kills all movement"""
    result = send_gcode("M112")
    return jsonify(result)

@app.route('/speak', methods=['POST'])  # FIXED: Added missing route decorator
@async_route
async def speak():
    """
    Endpoint to convert text to speech and play it
    Expected JSON payload: {"text": "text to speak", "voice": "coral" (optional)}
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'text' field in request body"
            }), 400
        
        text = data['text']
        voice = data.get('voice', 'coral')  # Default to coral voice
        instructions = data.get('instructions', 'Speak in a cheerful and positive tone.') 
        
        if not text.strip():
            return jsonify({
                "success": False,
                "error": "Text cannot be empty"
            }), 400
       
        try:
            # Use the correct client instance
            async with openai_client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice=voice,
                input=text,
                instructions=instructions,
                response_format="pcm",
            ) as response:
                await LocalAudioPlayer().play(response) 
            return jsonify({
                "success": True,
                "message": "Text spoken successfully",
                "text": text,
                "voice": voice
            }), 200
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Audio playback error: {str(e)}"
            }), 500
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "camera_available": check_camera_available(),
        "openai_configured": bool(os.getenv('OPENAI_API_KEY'))
    }), 200

def check_camera_available():
    """Check if camera is available"""
    try:
        result = subprocess.run(["rpicam-hello", "--list-cameras"], 
                              capture_output=True, text=True, timeout=5)
        return "No cameras available" not in result.stderr
    except:
        return False

@app.route('/', methods=['GET'])
def index():
    """API documentation"""
    return jsonify({
        "api_name": "Pi Camera & TTS Server",
        "version": "1.0",
        "endpoints": {
            "/take_picture": {
                "method": "POST",
                "description": "Take a picture with the Pi camera",
                "returns": "JSON with success status and download URL"
            },
            "/download_image/<filename>": {
                "method": "GET", 
                "description": "Download captured image"
            },
            "/speak": {
                "method": "POST",
                "description": "Convert text to speech and play it",
                "payload": {
                    "text": "Text to speak (required)",
                    "voice": "Voice to use: coral, sage, ash (optional, default: coral)",
                    "instructions": "Speaking instructions (optional)"
                }
            },
            "/printer/home": {
                "method": "POST",
                "description": "Home all printer axes"
            },
            "/printer/move": {
                "method": "POST", 
                "description": "Move printer to coordinates",
                "payload": {
                    "x": "X coordinate (optional)",
                    "y": "Y coordinate (optional)", 
                    "z": "Z coordinate (optional)",
                    "speed": "Movement speed in mm/min (optional)"
                }
            },
            "/printer/position": {
                "method": "GET",
                "description": "Get current printer position"
            },
            "/printer/status": {
                "method": "GET", 
                "description": "Get printer firmware info"
            },
            "/printer/command": {
                "method": "POST",
                "description": "Send raw G-code command",
                "payload": {"command": "G-code command to send"}
            },
            "/printer/emergency_stop": {
                "method": "POST",
                "description": "Emergency stop all printer movement"
            },
            "/health": {
                "method": "GET",
                "description": "Health check - shows camera and OpenAI status"
            }
        }
    })

if __name__ == '__main__':
    # Check dependencies on startup
    print("Starting Pi Camera & TTS Server...")
    print(f"Camera available: {check_camera_available()}")
    print(f"OpenAI API key configured: {bool(os.getenv('OPENAI_API_KEY'))}")
    
    # Run Flask server
    app.run(host='0.0.0.0', port=5000, debug=True)
