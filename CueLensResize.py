import ctypes
# Ensure high-DPI displays are handled correctly (physical pixels match logical)
ctypes.windll.user32.SetProcessDPIAware()

import cv2
import json
import base64
import asyncio
import websockets
import ssl
import numpy as np
import mss
import tkinter as tk
from tkinter import messagebox
import sys
import threading
from PIL import Image, ImageTk

# Hume WebSocket API URL and your API key
API_KEY = "8mXUfwdUs48Co3xoFKAMBeYlE2gDuzFBCirS3RB1GL8RLUhR"
WEBSOCKET_URL = "wss://api.hume.ai/v0/stream/models"

# Flag to control main loop
running = True

# Standard API dimensions - what the API expects
API_WIDTH = 1280
API_HEIGHT = 720

class ControlWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # Configure control window
        self.title("Emotion Detector")
        self.geometry("300x150")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self.quit_app)
        
        # Add controls and information
        header = tk.Label(self, text="Emotion Detector", font=("Arial", 14, "bold"))
        header.pack(pady=10)
        
        self.status_label = tk.Label(self, text="Status: Running")
        self.status_label.pack(pady=5)
        
        help_text = tk.Label(self, text="Click Exit to close")
        help_text.pack(pady=5)
        
        exit_button = tk.Button(self, text="Exit", command=self.quit_app)
        exit_button.pack(pady=10)
        
        # Store screen dimensions (now physical pixels)
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()
        
        # Create the transparent overlay
        self.create_overlay()
        
    def create_overlay(self):
        # Create overlay window
        self.overlay = tk.Toplevel(self)
        self.overlay.attributes('-alpha', 0.9)
        self.overlay.attributes('-topmost', True)
        self.overlay.attributes('-transparentcolor', 'black')
        self.overlay.overrideredirect(True)
        
        # Make it fullscreen
        self.overlay.geometry(f"{self.screen_width}x{self.screen_height}+0+0")
        
        # Create canvas
        self.canvas = tk.Canvas(self.overlay, 
                              width=self.screen_width,
                              height=self.screen_height,
                              highlightthickness=0,
                              bg='black')
        self.canvas.pack()
        
        # Store elements
        self.elements = []
        
        # Bind escape key
        self.overlay.bind('<Escape>', self.quit_app)
    
    def quit_app(self, event=None):
        global running
        running = False
        print("Quitting application...")
        self.destroy()

    def show_emotions(self, faces):
        # Clear previous elements
        for element in self.elements:
            self.canvas.delete(element)
        self.elements = []
        
        # Update status
        self.status_label.config(text=f"Status: Running - {len(faces)} faces detected")
        
        # Display faces and emotions
        for face in faces:
            x1, y1, x2, y2 = face["box"]
            emotions = face["emotions"]
            
            # Draw rectangle around face
            rect = self.canvas.create_rectangle(
                x1, y1, x2, y2, 
                outline='red', 
                width=3
            )
            self.elements.append(rect)
            
            # Add emotion label with white background
            if emotions:
                bg = self.canvas.create_rectangle(
                    x2 + 5, y1, 
                    x2 + 150, y1 + 40,
                    fill='white', 
                    outline='red',
                    width=2
                )
                self.elements.append(bg)
                text = f"{emotions[0]['name']}"
                label = self.canvas.create_text(
                    x2 + 15, y1 + 20,
                    text=text,
                    fill='blue',
                    anchor='w',
                    font=('Arial', 14, 'bold')
                )
                self.elements.append(label)
        
        self.update()

async def main():
    global running
    
    # Create control window (which also creates the overlay)
    app = ControlWindow()
    print("Emotion detection active. Press ESC or use the control window to quit.")
    
    # Screen capture
    with mss.mss() as sct:
        # Get the first monitor
        monitor = sct.monitors[1]
        
        # Store original capture dimensions
        capture_width = monitor["width"]
        capture_height = monitor["height"]
        print(f"Screen dimensions: {capture_width}x{capture_height}")
        
        # Setup secure connection
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        try:
            # Create websocket connection
            async with websockets.connect(
                WEBSOCKET_URL,
                extra_headers={"X-Hume-Api-Key": API_KEY},
                ssl=ssl_context
            ) as websocket:
                print("Connected to Hume API")
                
                frame_count = 0
                
                # Main loop
                while running and app.winfo_exists():
                    try:
                        app.update()
                        frame_count += 1
                        if frame_count % 15 == 0:
                            screenshot = sct.grab(monitor)
                            img = np.array(screenshot)
                            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                            original_height, original_width = img.shape[:2]
                            h_factor = API_HEIGHT / original_height
                            w_factor = API_WIDTH / original_width
                            resize_factor = min(h_factor, w_factor)
                            new_width = int(original_width * resize_factor)
                            new_height = int(original_height * resize_factor)
                            resized_img = cv2.resize(img, (new_width, new_height))
                            api_img = np.zeros((API_HEIGHT, API_WIDTH, 3), dtype=np.uint8)
                            x_offset = (API_WIDTH - new_width) // 2
                            y_offset = (API_HEIGHT - new_height) // 2
                            api_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_img
                            scaling = {
                                "factor": resize_factor,
                                "x_offset": x_offset,
                                "y_offset": y_offset
                            }
                            _, buffer = cv2.imencode('.jpg', api_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            img_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
                            message = {
                                "models": {"face": {}},
                                "data": img_base64
                            }
                            await websocket.send(json.dumps(message))
                            result = json.loads(await websocket.recv())
                            faces = []
                            for face_pred in result.get("face", {}).get("predictions", []):
                                bbox = face_pred.get("bbox")
                                emotions = face_pred.get("emotions")
                                if not bbox or not emotions:
                                    continue
                                api_x, api_y, api_w, api_h = map(float, (bbox["x"], bbox["y"], bbox["w"], bbox["h"]))
                                adj_x = api_x - scaling["x_offset"]
                                adj_y = api_y - scaling["y_offset"]
                                screen_x = int(adj_x / scaling["factor"])
                                screen_y = int(adj_y / scaling["factor"])
                                screen_w = int(api_w / scaling["factor"])
                                screen_h = int(api_h / scaling["factor"])
                                x1 = max(0, screen_x)
                                y1 = max(0, screen_y)
                                x2 = min(app.screen_width, screen_x + screen_w)
                                y2 = min(app.screen_height, screen_y + screen_h)
                                if x1 < x2 and y1 < y2:
                                    top3 = sorted(emotions, key=lambda e: e["score"], reverse=True)[:3]
                                    faces.append({"box": (x1, y1, x2, y2), "emotions": top3})
                            app.show_emotions(faces)
                        await asyncio.sleep(0.033)
                    except Exception as e:
                        print(f"Error: {e}")
                        await asyncio.sleep(1)
        except Exception as e:
            print(f"Connection error: {e}")
            messagebox.showerror("Connection Error", f"Could not connect to the Hume API: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program terminated by user")
    except Exception as e:
        print(f"Program terminated with error: {e}")
