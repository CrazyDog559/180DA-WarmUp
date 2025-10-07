# [180 DA, Lab 1] Python program to save a
# video using OpenCV


import cv2
import numpy

# Create an object to read
# from camera
video = cv2.VideoCapture(0,cv2.CAP_AVFOUNDATION)

# We need to check if camera
# is opened previously or not
if (video.isOpened() == False):
    print("Error reading video file")
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(video.get(3))
frame_height = int(video.get(4))

size = (frame_width, frame_height)

# Below VideoWriter object will create
# a frame of above defined The output
# is stored in 'filename.avi' file.
result = cv2.VideoWriter('test_video.avi',
                        cv2.VideoWriter_fourcc(*'MJPG'),
                        10, size)

print("Recording... Press S on keyboard to stop")

while(True):
    ret, frame = video.read()

    if ret == True:

        # Write the frame into the
        # file 'filename.avi'
        result.write(frame)

        # Display the frame
        # saved in the file
        cv2.imshow('Frame', frame)

        # Press S on keyboard to stop the process
        # THE FOCUS SHOULD BE ON THE RECORDING AND NOT THE TERMINAL
        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    # Break the loop
    else:
        break

# When everything done, release
# the video capture and video
# write objects
video.release()
result.release()
    
# Closes all the frames
cv2.destroyAllWindows()
cv2.waitKey(1)
print("The video was successfully saved")
