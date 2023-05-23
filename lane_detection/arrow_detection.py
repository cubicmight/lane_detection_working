# Python program to illustrate
# template matching
import cv2
import numpy as np

# Read the main image
img_rgb = cv2.imread('../input/IMG-8392.jpg')
# Convert it to grayscale
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# Read the template
template = cv2.imread('../input/IMG-8391.jpg', 0)

# Store width and height of template in w and h
w, h = template.shape[::-1]

# Perform match operations.
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

# Specify a threshold
threshold = 0.5

# Store the coordinates of matched area in a numpy array
loc = np.where(res >= threshold)

# Draw a rectangle around the matched region.
for pt in zip(*loc[::-1]):
	cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

# Show the final image with the matched area.
cv2.imshow('Detected', img_rgb)
cv2.waitKey(50000)

def detect():
            while video.isOpened():
                # Read the current frame
                ret, frame = video.read()

                if not ret:
                    break

                # Convert the frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                detected_images = []

                for template, template_size, template_name in zip(templates, template_sizes, template_names):
                    # Perform template matching
                    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)

                    # Find matches above the threshold
                    locations = np.where(result >= threshold)

                    # Draw bounding boxes around the matched regions
                    for loc in zip(*locations[::-1]):
                        cv2.rectangle(frame, loc, (loc[0] + template_size[1], loc[1] + template_size[0]), (0, 255, 0), 2)
                        detected_images.append(template_name)

                # Display the resulting frame with detected words
                for i, detected_image in enumerate(detected_images):
                    if detected_image == 'rightTurnArrow.png':
                        cv2.putText(frame, 'right', (10, 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    elif detected_image == 'leftTurnArrow.png':
                        cv2.putText(frame, 'left', (10, 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Convert the frame to JPEG format
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                # Yield the frame for displaying in the HTML page
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
