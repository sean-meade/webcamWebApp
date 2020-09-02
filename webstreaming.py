# import the necessary packages
from pyimagesearch.motion_detection import SingleMotionDetector 
# The VideoStream class (Line 3) will enable us to access our Raspberry Pi camera module 
# or USB webcam.
import imutils
from imutils.video import VideoStream
# The following 2 handle importing our required Flask packages — we’ll be using these 
# packages to render our index.html template and serve it up to clients.
from flask import Response
from flask import Flask
from flask import render_template
# threading library to ensure we can support concurrency (i.e., multiple clients, web 
# browsers, and tabs at the same time).
import threading
import argparse
import datetime
import imutils
import time
import cv2


"""
Let’s move on to performing a few initializations:
"""
# initialize the output frame and a lock used to ensure thread-safe exchanges of the output 
# frames (useful when multiple browsers/tabs are viewing the stream/(i.e., ensuring that one 
# thread isn’t trying to read the frame as it is being updated).)
"""
First, we initialize our outputFrame — this will be the frame (post-motion 
detection) that will be served to the clients.
"""
outputFrame = None
lock = threading.Lock()

# initialize a flask object
"""initialize our Flask app itself"""
app = Flask(__name__)

# initialize/access the video stream and allow the camera sensor to warm up
# vs = VideoStream(usePiCamera = 1).start()
"""
*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!
Possibly where to insert webcam: the src that follows
!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*
"""
vs = VideoStream(src = 0).start()
time.sleep(2.0)

"""The next function, index, will render our index.html template and serve 
up the output video stream:

This function is quite simplistic — all it’s doing is calling the Flask render_template 
on our HTML file.

"""
@app.route("/")
def index():
    #return the rendered template
    return render_template("index.html")

"""
Our next function is responsible for:

1) Looping over frames from our video stream
2) Applying motion detection
3) Drawing any results on the outputFrame

And furthermore, this function must perform all of these operations in a thread safe 
manner to ensure concurrency is supported.

Our detection_motion function accepts a single argument, frameCount, which is the minimum 
number of required frames to build our background bg in the SingleMotionDetector class:

- If we don’t have at least frameCount frames, we’ll continue to compute the accumulated 
    weighted average.
- Once frameCount is reached, we’ll start performing background subtraction.
"""
def detect_motion(frameCount):
    # grab global references to the video stream, output frame, and lock variables
    """
    grabs global references to three variables:

    -    vs: Our instantiated VideoStream object
    -    outputFrame: The output frame that will be served to clients
    -    lock: The thread lock that we must obtain before updating outputFrame
    """
    global vs, outputFrame, lock

    # initialize the motion detector and the total number of frames read thus far
    """
    This initializes our SingleMotionDetector class with a value of accumWeight=0.1, implying that 
    the bg value will be weighted higher when computing the weighted average
    """
    md = SingleMotionDetector.SingleMotionDetector(accumWeight = 0.1)
    """
    This then initializes the total number of frames read thus far — we’ll need to ensure a 
    sufficient number of frames have been read to build our background model.
    """
    total = 0

    """
    From there, we’ll be able to perform background subtraction.

    With these initializations complete, we can now start looping over frames 
    from the camera:
    """

    # Loop over frames from the video stream
    while True:
        # read the next frame from the video stream, resize it, convert the frame to grayscale,
        # and blur it
        """reads the next frame from our camera"""
        frame = vs.read()
        """Resizing to have a width of 400px (the smaller our input frame is, the less data there 
        is, and thus the faster our algorithms will run)."""
        frame = imutils.resize(frame, width = 400)
        """Converting to grayscale"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        """Gaussian blurring (to reduce noise)"""
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        #grab the current timestamp and draw it on the frame
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M%S%p"), (10, frame.shape[0] - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        
        """With one final check, we can perform motion detection:"""
        # if the total number of frames has reached a sufficient number to construct a 
        # reasonable background model, then continue to process the frame
        """we ensure that we have read at least frameCount frames 
            to build our background subtraction model."""
        if total > frameCount:
            # detect motion in the image
            """If so, we apply the .detect motion of our motion 
                detector, which returns a single variable, motion."""
            motion = md.detect(gray)

            # check to see if motion was found in the frame
            """If motion is None, then we know no motion has taken 
                place in the current frame. Otherwise, if motion is 
                not None, then we need to draw the bounding 
                box coordinates of the motion region on the frame."""
            if motion is not None:
                # unpack the tuple and draw the box surrounding the "motion area" on the 
                # output frame
                (thresh, (minX, minY, maxX, maxY)) = motion
                cv2.rectangle(frame, (minX, minY), (maxX, maxY), 
                (0, 0, 255), 2)

        # update the background model and increment the total number of frames read thus far
        """updates our motion detection background model"""
        md.update(gray)
        """increments the total number of frames read from the camera thus far."""
        total += 1

        #acquire the lock, set the output frame, and release the lock
        """acquires the lock required to support thread concurrency"""
        with lock:
            """sets the outputFrame"""
            outputFrame = frame.copy()  

"""
We need to acquire the lock to ensure the outputFrame variable is not accidentally being 
read by a client while we are trying to update it.
"""

"""Our next function, generate , is a Python generator used to 
    encode our outputFrame as JPEG data"""

def generate():
    # grab global references to the output frame and lock variables
    """grabs global references to our outputFrame and lock, similar to the detect_motion function."""
    global outputFrame, lock

    # loop over frames from the output stream
    """Then generate starts an infinite loop that will ontinue until we kill the script."""
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip the iteration of the loop
            """Ensure the outputFrame is not empty, which may happen if a frame is 
                dropped from the camera sensor."""
            if outputFrame is None:
                continue
            
            # Encode the frame in JPEG format
            """Encode the frame as a JPEG image - JPEG compression is performed here 
                to reduce load on the network and ensure faster transmission of frames."""
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            """Check to see if the success flag has failed implying that the JPEG 
                compression failed and we should ignore the frame."""
            if not flag:
                continue

        # yield the output frame in the byte format
        """Finally, serve the encoded JPEG frame as a byte array that can be 
            consumed by a web browser."""
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')

"""Notice how this function as the app.route signature, just like the index function above.
    The app.route signature tells Flask that this function is a URL endpoint and that data 
    is being served from http://your_ip_address/video_feed."""
@app.route("/video_feed")
def video_feed():
    # return the response generated along with specific media type (mime type)
    """The output of video_feed is the live motion detection output, encoded as a byte array 
        via the generate function. Your web browser is smart enough to take this byte array 
        and display it in your browser as a live feed."""
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary = frame")

"""Our final code block handles parsing command line arguments and launching the Flask app:"""
# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    """--ip: The IP address of the system you are launching the webstream.py file from"""
    ap.add_argument("-i", "--ip", type = str, required = True, help = "ip address of the device")
    """--port: The port number that the Flask app will run on (you’ll typically supply a 
        value of 8000 for this parameter)"""
    ap.add_argument("-o", "--port", type = int, required = True, help = "ephemeral port number of the server (1024 to 65535")
    """--frame-count: The number of frames used to accumulate and build the background model 
        before motion detection is performed. By default, we use 32  frames to build the 
        background model"""
    ap.add_argument("-f", "--frame-count", type = int, default = 32, help = "# of frames used to construct the background model")
    args = vars(ap.parse_args())

    # start a thread that will perform motion detection
    """launch a thread that will be used to perform motion detection.

        Using a thread ensures the detect_motion function can safely 
        run in the background — it will be constantly running and 
        updating our outputFrame so we can serve any motion detection 
        results to our clients.
    """
    t = threading.Thread(target = detect_motion, args = (args["frame_count"],))
    t.daemon = True
    """launches the Flask app itself"""
    t.start()

    # start the flask app
    app.run(host = args["ip"], port = args["port"], debug = True, threaded = True, use_reloader = False)

# release the video stream pointer
vs.stop()