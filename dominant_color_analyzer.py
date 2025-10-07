"""
[180DA Lab] Enhanced K-Means Dominant Color Analysis with Brightness Robustness Testing

BASELINE CODE SOURCES:
- kmeans.py from course materials (basic K-means color clustering on static images)
- Scikit-learn KMeans: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- OpenCV K-means tutorial: https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time

class DominantColorAnalyzer:
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        self.n_colors = 3  # Number of dominant colors to find
        self.roi_size = 150  # Size of central rectangle
        self.recording_results = False
        
        # For brightness testing
        self.brightness_results = []
        
    def extract_colors(self, image, n_colors=3):
        """Extract dominant colors using K-means"""
        # Reshape image to be a list of pixels
        data = image.reshape((-1, 3))
        data = np.float32(data)
        
        # Apply K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to uint8 and find percentages
        centers = np.uint8(centers)
        
        # Count labels to find percentages
        unique_labels, counts = np.unique(labels, return_counts=True)
        percentages = counts / len(labels)
        
        # Sort by percentage (most dominant first)
        sorted_indices = np.argsort(percentages)[::-1]
        
        return centers[sorted_indices], percentages[sorted_indices]
    
    def create_color_bar(self, colors, percentages, width=300, height=60):
        """Create a color bar showing dominant colors and their percentages"""
        bar = np.zeros((height, width, 3), dtype=np.uint8)
        
        start_x = 0
        for i, (color, percentage) in enumerate(zip(colors, percentages)):
            end_x = start_x + int(percentage * width)
            bar[:, start_x:end_x] = color
            
            # Add percentage text
            text_x = start_x + (end_x - start_x) // 2 - 20
            cv2.putText(bar, f"{percentage:.1%}", (max(5, text_x), height//2 + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            start_x = end_x
        
        return bar
    
    def calculate_brightness(self, image):
        """Calculate average brightness of image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
    
    def color_distance(self, color1, color2):
        """Calculate Euclidean distance between two colors"""
        return np.sqrt(np.sum((color1 - color2) ** 2))
    
    def analyze_color_stability(self, colors_history):
        """Analyze how stable the dominant colors are over time"""
        if len(colors_history) < 2:
            return 0
        
        # Compare most recent colors with previous ones
        recent_colors = colors_history[-1]
        previous_colors = colors_history[-2]
        
        # Find minimum distance between color sets
        min_distances = []
        for recent_color in recent_colors:
            distances = [self.color_distance(recent_color, prev_color) 
                        for prev_color in previous_colors]
            min_distances.append(min(distances))
        
        avg_distance = np.mean(min_distances)
        stability = max(0, 100 - avg_distance)  # Convert to stability score 0-100
        return stability
    
    def run_analysis(self):
        """Main analysis loop"""
        print("Dominant Color Analyzer")
        print("Instructions:")
        print("- A central rectangle shows the analysis region")
        print("- Press 'r' to start/stop recording brightness test")
        print("- Press '+'/'-' to change rectangle size")
        print("- Press 'v' to start/stop video recording")
        print("- Press 'c' to clear results and history")
        print("- Press 's' to stop")
        
        # Video recording setup
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))
        size = (frame_width, frame_height)
        
        video_writer = None
        recording_video = False
        
        colors_history = []
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            height, width = frame.shape[:2]
            
            # Define central rectangle (ROI - Region of Interest)
            center_x, center_y = width // 2, height // 2
            x1 = center_x - self.roi_size // 2
            y1 = center_y - self.roi_size // 2
            x2 = center_x + self.roi_size // 2
            y2 = center_y + self.roi_size // 2
            
            # Extract ROI
            roi = frame[y1:y2, x1:x2]
            
            # Analyze dominant colors every few frames (for performance)
            if frame_count % 5 == 0:
                try:
                    dominant_colors, percentages = self.extract_colors(roi, self.n_colors)
                    colors_history.append(dominant_colors)
                    
                    # Keep only recent history
                    if len(colors_history) > 10:
                        colors_history.pop(0)
                    
                    # Calculate brightness and stability
                    brightness = self.calculate_brightness(roi)
                    stability = self.analyze_color_stability(colors_history)
                    
                    # Create color bar
                    color_bar = self.create_color_bar(dominant_colors, percentages)
                    
                    # Record results if in recording mode
                    if self.recording_results:
                        result = {
                            'time': time.strftime("%H:%M:%S"),
                            'brightness': brightness,
                            'stability': stability,
                            'dominant_color': dominant_colors[0].tolist(),
                            'dominant_percentage': percentages[0]
                        }
                        self.brightness_results.append(result)
                    
                except Exception as e:
                    print(f"Analysis error: {e}")
                    dominant_colors = np.array([[128, 128, 128], [64, 64, 64], [192, 192, 192]])
                    percentages = np.array([0.5, 0.3, 0.2])
                    color_bar = np.zeros((60, 300, 3), dtype=np.uint8)
                    brightness = 0
                    stability = 0
            
            # Draw ROI rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Analysis Region", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Display information
            y_offset = 30
            
            cv2.putText(frame, f"Brightness: {brightness:.1f}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            
            cv2.putText(frame, f"Stability: {stability:.1f}%", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            
            if self.recording_results:
                cv2.putText(frame, "RECORDING", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                y_offset += 20
            
            # Display dominant colors
            try:
                for i, (color, percentage) in enumerate(zip(dominant_colors, percentages)):
                    color_bgr = [int(color[2]), int(color[1]), int(color[0])]  # Convert RGB to BGR
                    text = f"Color {i+1}: {percentage:.1%} RGB({color[0]},{color[1]},{color[2]})"
                    cv2.putText(frame, text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 1)
                    y_offset += 20
            except:
                pass
            
            # Show color bar at bottom
            try:
                bar_y = height - 70
                frame[bar_y:bar_y+60, 10:310] = color_bar
            except:
                pass
            
            # Write frame to video if recording
            if recording_video and video_writer is not None:
                video_writer.write(frame)
            
            # Add video recording indicator
            if recording_video:
                cv2.putText(frame, "RECORDING VIDEO", (width - 200, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow('Dominant Color Analysis', frame)
            cv2.imshow('ROI', roi)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                break
            elif key == ord('r'):
                self.recording_results = not self.recording_results
                if self.recording_results:
                    print("Started recording test results")
                else:
                    print("Stopped recording")
            elif key == ord('+') or key == ord('='):
                self.roi_size = min(300, self.roi_size + 20)
                print(f"ROI size: {self.roi_size}")
            elif key == ord('-'):
                self.roi_size = max(50, self.roi_size - 20)
                print(f"ROI size: {self.roi_size}")
            elif key == ord('c'):
                # Clear results
                self.brightness_results = []
                colors_history = []
                print("Cleared results and history")
            elif key == ord('v'):
                # Toggle video recording
                if not recording_video:
                    # Start recording
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f'dominant_color_analysis_{timestamp}.avi'
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
            
            frame_count += 1
        
        # Print results summary
        self.print_results_summary()
        
        # Clean up video recording
        if video_writer is not None:
            video_writer.release()
            print("Video recording stopped and saved")
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def print_results_summary(self):
        """Print summary of recorded analysis data"""
        if not self.brightness_results:
            print("No analysis data recorded.")
            return
        
        print("\n=== DOMINANT COLOR ANALYSIS RESULTS ===")
        print(f"Total samples recorded: {len(self.brightness_results)}")
        
        # Calculate statistics for all recorded data
        brightnesses = [r['brightness'] for r in self.brightness_results]
        stabilities = [r['stability'] for r in self.brightness_results]
        
        print(f"\nBrightness Statistics:")
        print(f"  Min: {min(brightnesses):.1f}")
        print(f"  Max: {max(brightnesses):.1f}")
        print(f"  Average: {np.mean(brightnesses):.1f}")
        
        print(f"\nStability Statistics:")
        print(f"  Min: {min(stabilities):.1f}%")
        print(f"  Max: {max(stabilities):.1f}%")
        print(f"  Average: {np.mean(stabilities):.1f}%")
        
        # Most common dominant color
        dominant_colors = [tuple(r['dominant_color']) for r in self.brightness_results]
        from collections import Counter
        most_common = Counter(dominant_colors).most_common(3)  # Show top 3
        
        print(f"\nMost Common Dominant Colors:")
        for i, (color, count) in enumerate(most_common, 1):
            percentage = (count / len(self.brightness_results)) * 100
            print(f"  {i}. RGB{color} - {count} times ({percentage:.1f}%)")
        
        # Show recent samples
        print(f"\nRecent Samples (last 5):")
        for result in self.brightness_results[-5:]:
            print(f"  {result['time']}: Brightness={result['brightness']:.1f}, Stability={result['stability']:.1f}%, Color=RGB{tuple(result['dominant_color'])}")

if __name__ == "__main__":
    analyzer = DominantColorAnalyzer()
    analyzer.run_analysis()