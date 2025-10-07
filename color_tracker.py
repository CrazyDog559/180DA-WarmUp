"""
[180DA Lab] Enhanced Color Tracking with Bounding Box

BASELINE CODE SOURCES:
- Original threshold.py from course materials (HSV blue color tracking)
- OpenCV color detection tutorials: https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
- Mouse callback examples: https://docs.opencv.org/4.x/db/d5b/tutorial_py_mouse_handling.html

"""

import cv2
import numpy as np

class ColorTracker:
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        self.mode = 'HSV'  # Start with HSV mode
        self.calibrated = False
        
        # Store the last clicked color values
        self.last_hsv_color = None
        self.last_rgb_color = None
        
        # Store click position and target object info
        self.click_x = None
        self.click_y = None
        self.target_contour = None
        self.target_center = None
        
        # Default color ranges (you can adjust these)
        self.hsv_lower = np.array([110, 50, 50])   # Default blue in HSV
        self.hsv_upper = np.array([130, 255, 255])
        
        self.rgb_lower = np.array([0, 0, 100])     # Default blue in RGB
        self.rgb_upper = np.array([100, 100, 255])
        
        # Threshold ranges - you can experiment with these
        self.hsv_range = [15, 50, 50]  # H, S, V ranges - reduced for better tracking
        self.rgb_range = [30, 30, 30]    # R, G, B ranges - reduced for better tracking
        
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback to select color for calibration"""
        if event == cv2.EVENT_LBUTTONDOWN:
            frame = param
            
            # Store click position for object identification
            self.click_x = x
            self.click_y = y
            
            if self.mode == 'HSV':
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                color = hsv[y, x]
                self.last_hsv_color = color.copy()  # Store the clicked color
                
                self.update_hsv_ranges()
                print(f"HSV Color selected at ({x},{y}): H={color[0]}, S={color[1]}, V={color[2]}")
                print(f"HSV Range: Lower {self.hsv_lower}, Upper {self.hsv_upper}")
                
            else:  # RGB mode
                color = frame[y, x]
                self.last_rgb_color = color.copy()  # Store the clicked color
                
                self.update_rgb_ranges()
                print(f"RGB Color selected at ({x},{y}): R={color[2]}, G={color[1]}, B={color[0]}")
                print(f"RGB Range: Lower {self.rgb_lower}, Upper {self.rgb_upper}")
            
            self.calibrated = True
            
            # Draw a small circle to show where user clicked
            cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
            print(f"Object selected at position ({x}, {y}) - Now tracking this specific object")
    
    def update_hsv_ranges(self):
        # Update HSV color ranges based on last clicked color and current threshold
        if self.last_hsv_color is not None:
            color = self.last_hsv_color
            self.hsv_lower = np.array([
                max(0, color[0] - self.hsv_range[0]),
                max(0, color[1] - self.hsv_range[1]),
                max(0, color[2] - self.hsv_range[2])
            ])
            self.hsv_upper = np.array([
                min(179, color[0] + self.hsv_range[0]),
                min(255, color[1] + self.hsv_range[1]),
                min(255, color[2] + self.hsv_range[2])
            ])
    
    def update_rgb_ranges(self):
        # Update RGB color ranges based on last clicked color and current threshold
        if self.last_rgb_color is not None:
            color = self.last_rgb_color
            self.rgb_lower = np.array([
                max(0, color[2] - self.rgb_range[0]),  # B
                max(0, color[1] - self.rgb_range[1]),  # G
                max(0, color[0] - self.rgb_range[2])   # R
            ])
            self.rgb_upper = np.array([
                min(255, color[2] + self.rgb_range[0]),  # B
                min(255, color[1] + self.rgb_range[1]),  # G
                min(255, color[0] + self.rgb_range[2])   # R
            ])
    
    def find_target_object(self, contours):
        # Find the contour that contains or is closest to the clicked point
        if self.click_x is None or self.click_y is None:
            return None
            
        click_point = (self.click_x, self.click_y)
        
        # First, try to find a contour that contains the click point
        for contour in contours:
            if cv2.pointPolygonTest(contour, click_point, False) >= 0:
                return contour
        
        # If no contour contains the point, find the closest one
        min_distance = float('inf')
        closest_contour = None
        
        for contour in contours:
            # Calculate center of contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Calculate distance from click point to contour center
                distance = np.sqrt((cx - self.click_x)**2 + (cy - self.click_y)**2)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_contour = contour
        
        return closest_contour

    def track_color(self):
        """Main tracking loop"""
        cv2.namedWindow('Color Tracker')
        cv2.setMouseCallback('Color Tracker', self.mouse_callback)
        
        print("Color Tracker Started!")
        print("Controls:")
        print("- Press 'h' for HSV mode")
        print("- Press 'r' for RGB mode")
        print("- Press 'c' then click on object to calibrate color")
        print("- Press 'v' to start/stop video recording")
        print("- Press '+'/'-' to adjust threshold ranges")
        print("- Press 's' to stop")
        
        # Video recording setup
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))
        size = (frame_width, frame_height)
        
        video_writer = None
        recording_video = False
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Set mouse callback parameter to current frame
            cv2.setMouseCallback('Color Tracker', self.mouse_callback, frame)
            
            if self.mode == 'HSV':
                # HSV thresholding
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
                color_info = f"HSV Mode - Range: {self.hsv_range}"
            else:
                # RGB thresholding
                mask = cv2.inRange(frame, self.rgb_lower, self.rgb_upper)
                color_info = f"RGB Mode - Range: {self.rgb_range}"
            
            # Find contours for color detection
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area to remove noise
            min_area = 100
            filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            
            detection_count = len(filtered_contours)
            tracking_success = False
            
            if self.calibrated and filtered_contours:
                # Find the target object (the one we clicked on)
                target_contour = self.find_target_object(filtered_contours)
                
                if target_contour is not None:
                    # Draw bounding box around the target object
                    x, y, w, h = cv2.boundingRect(target_contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    
                    # Calculate and store center for tracking continuity
                    M = cv2.moments(target_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        self.target_center = (cx, cy)
                        
                        # Draw center point
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                        
                        # Update click position to follow the object
                        self.click_x = cx
                        self.click_y = cy
                    
                    # Add tracking info
                    area = cv2.contourArea(target_contour)
                    cv2.putText(frame, f"TRACKING - Area: {int(area)}", (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    tracking_success = True
                
                # Draw all detected contours in light color for reference
                for contour in filtered_contours:
                    if contour is not target_contour:  # Don't redraw the target
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)
            
            # Add mode and instructions text
            cv2.putText(frame, color_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show calibration and tracking status
            if self.calibrated:
                if tracking_success:
                    status_text = f"TRACKING SUCCESS - {detection_count} objects detected"
                    status_color = (0, 255, 0)
                else:
                    status_text = f"LOST TARGET - {detection_count} objects detected"
                    status_color = (0, 165, 255)  # Orange
            else:
                status_text = "NOT CALIBRATED - Click on object to track"
                status_color = (0, 0, 255)
            
            cv2.putText(frame, status_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            
            # Show current color values if calibrated
            if self.calibrated:
                if self.mode == 'HSV' and self.last_hsv_color is not None:
                    color_text = f"Target HSV: {self.last_hsv_color}"
                elif self.mode == 'RGB' and self.last_rgb_color is not None:
                    color_text = f"Target RGB: {self.last_rgb_color}"
                else:
                    color_text = "No target color"
                cv2.putText(frame, color_text, (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Write frame to video if recording
            if recording_video and video_writer is not None:
                video_writer.write(frame)
            
            # Add video recording indicator
            if recording_video:
                cv2.putText(frame, "RECORDING VIDEO", (frame_width - 200, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Show original frame and mask
            cv2.imshow('Color Tracker', frame)
            cv2.imshow('Mask', mask)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                break
            elif key == ord('h'):
                self.mode = 'HSV'
                print("Switched to HSV mode")
            elif key == ord('r'):
                self.mode = 'RGB'
                print("Switched to RGB mode")
            elif key == ord('c'):
                print("Click on the object you want to track")
            # Adjust threshold ranges with +/- keys
            elif key == ord('+') or key == ord('='):
                if self.mode == 'HSV':
                    self.hsv_range = [r + 5 for r in self.hsv_range]
                    self.update_hsv_ranges()  # Update ranges with new threshold
                    print(f"Increased HSV range to: {self.hsv_range}")
                else:
                    self.rgb_range = [r + 5 for r in self.rgb_range]
                    self.update_rgb_ranges()  # Update ranges with new threshold
                    print(f"Increased RGB range to: {self.rgb_range}")
            elif key == ord('-'):
                if self.mode == 'HSV':
                    self.hsv_range = [max(5, r - 5) for r in self.hsv_range]
                    self.update_hsv_ranges()  # Update ranges with new threshold
                    print(f"Decreased HSV range to: {self.hsv_range}")
                else:
                    self.rgb_range = [max(5, r - 5) for r in self.rgb_range]
                    self.update_rgb_ranges()  # Update ranges with new threshold
                    print(f"Decreased RGB range to: {self.rgb_range}")
            elif key == ord('v'):
                # Toggle video recording
                if not recording_video:
                    # Start recording
                    import time
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f'color_tracking_{timestamp}.avi'
                    video_writer = cv2.VideoWriter(filename,
                                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                                 10, size)
                    recording_video = True
                    print(f"Started recording video: {filename}")
                else:
                    # Stop recording
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                    recording_video = False
                    print("Stopped recording video")
        
        # Clean up video recording
        if video_writer is not None:
            video_writer.release()
            print("Video recording stopped and saved")
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = ColorTracker()
    tracker.track_color()