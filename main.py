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
from PIL import Image, ImageDraw
import io

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

def add_grid_overlay(image_path, grid_size=20, grid_color=(255, 255, 255, 128)):
    """
    Add grid overlay to image using PIL
    Args:
        image_path: Path to the image file
        grid_size: Size of grid squares in pixels (default 20)
        grid_color: RGBA color tuple for grid lines (default semi-transparent white)
    Returns:
        PIL Image object with grid overlay
    """
    # Open the image
    image = Image.open(image_path)
    
    # Convert to RGBA if not already (for transparency support)
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Create a transparent overlay
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    width, height = image.size
    
    # Draw vertical lines
    for x in range(0, width + 1, grid_size):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=2)
    
    # Draw horizontal lines
    for y in range(0, height + 1, grid_size):
        draw.line([(0, y), (width, y)], fill=grid_color, width=2)
    
    # Composite the overlay onto the original image
    result = Image.alpha_composite(image, overlay)
    
    # Convert back to RGB if needed (for JPEG output)
    if result.mode == 'RGBA':
        result = result.convert('RGB')
    
    return result

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
                "download_url": f"/download_image/{filename}",
                "download_url_with_grid": f"/download_image/{filename}?grid=true"
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
    Query parameters:
        grid=true : Add 20px grid overlay to the image
        grid_size=N : Custom grid size in pixels (default 20)
        grid_color=RRGGBB : Grid color in hex format (default FFFFFF)
        grid_opacity=N : Grid opacity 0-255 (default 128)
    """
    try:
        filepath = os.path.join("/tmp", filename)
        if not os.path.exists(filepath):
            return jsonify({"error": "File not found"}), 404
        
        # Check if grid overlay is requested
        add_grid = request.args.get('grid', 'false').lower() == 'true'
        
        if add_grid:
            # Get grid parameters from query string
            grid_size = int(request.args.get('grid_size', 20))
            grid_color_hex = request.args.get('grid_color', 'FFFFFF')
            grid_opacity = int(request.args.get('grid_opacity', 128))
            
            # Convert hex color to RGB
            try:
                grid_color = tuple(int(grid_color_hex[i:i+2], 16) for i in (0, 2, 4))
                grid_color = (*grid_color, grid_opacity)  # Add alpha channel
            except:
                grid_color = (255, 255, 255, 128)  # Default white with transparency
            
            # Create image with grid overlay
            image_with_grid = add_grid_overlay(filepath, grid_size, grid_color)
            
            # Save to memory buffer
            img_buffer = io.BytesIO()
            image_with_grid.save(img_buffer, format='JPEG', quality=95)
            img_buffer.seek(0)
            
            # Create new filename for grid version
            name, ext = os.path.splitext(filename)
            grid_filename = f"{name}_grid{ext}"
            
            return send_file(
                img_buffer,
                mimetype='image/jpeg',
                as_attachment=True,
                download_name=grid_filename
            )
        else:
            # Return original image without grid
            return send_file(filepath, as_attachment=True, download_name=filename)
            
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
                "description": "Download captured image",
                "query_parameters": {
                    "grid": "Set to 'true' to add grid overlay (default: false)",
                    "grid_size": "Grid size in pixels (default: 20)",
                    "grid_color": "Grid color in hex format RRGGBB (default: FFFFFF)",
                    "grid_opacity": "Grid opacity 0-255 (default: 128)"
                },
                "examples": [
                    "/download_image/photo_123.jpg",
                    "/download_image/photo_123.jpg?grid=true",
                    "/download_image/photo_123.jpg?grid=true&grid_size=30&grid_color=FF0000&grid_opacity=200"
                ]
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
