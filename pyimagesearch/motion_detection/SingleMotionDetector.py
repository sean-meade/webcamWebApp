# Import the packages needed
import numpy as np
import imutils
import cv2

# Notes and code taken from: https://www.pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/


# Create class for motion detection
# This class accepts optional argument: accumWeight
# accumWeight is the factor used to our accumulated weighted average
"""
The larger accumWeight is, the less the background (bg) will be factored in when 
accumulating the weighted average.

Conversely, the smaller accumWeight is, the more the background bg will be considered 
when computing the average.

The accumWeight is set to 0.5 to make both the background and foreground even. Experimentation
can be done though.
"""
class SingleMotionDetector:
    # Create initial method
    def __init__(self, accumWeight = 0.5):
        # Store the accumulated weight factor
        self.accumWeight = accumWeight

        # initialize the background model
        self.bg = None
    
    # Create the update function. This takes in the frame (or image) and computes the 
    # weighted average. 
    def update(self, image):
        # if the background model is None, initialize it
        """In the case that our bg frame is None (implying that update has never been called), 
            we simply store the bg frame (Lines 15-18).
        """
        if self.bg is None:
            self.bg = image.copy().astype("float")
            return

        # update the background model by accumulating the weighted average
        # cv2.accumulateWeighted: https://docs.opencv.org/2.4/modules/imgproc/doc/motion_analysis_and_object_tracking.html#accumulateweighted
        """
        Otherwise, we compute the weighted average between the input frame, the existing 
        background bg, and our corresponding accumWeight factor.
        """
        cv2.accumulateWeighted(image, self.bg, self.accumWeight)

    """
    The detect method requires a single parameter along with an optional one:
        - image: The input frame/image that motion detection will be applied to.
        - tVal: The threshold value used to mark a particular pixel as “motion” or not.

    Given our background bg we can now apply motion detection via the detect method:
    """
    def detect(self, image, tVal = 25):
        # compute the absolute difference between the background model and the image
        # passed in, then thresholdth delta image
        """Given our input image we compute the absolute difference between the 
        image and the bg:"""
        delta = cv2.absdiff(self.bg.astype("unit8"), image)
        """
        Any pixel locations that have a difference > tVal are set to 255 (white; foreground), 
        otherwise they are set to 0 (black; background):
        """
        thresh = cv2.threhold(delta, tVal, 255, cv2.THRESH_BINARY)[1]

        # perform a series of erosions and dilations to remove small blobs
        """
        A series of erosions and dilations are performed to remove noise and small, localized 
        areas of motion that would otherwise be considered false-positives (like:
        """
        thresh = cv2.erode(thresh, None, iteration = 2)
        thresh = cv2.dilate(thresh, None, iterations = 2)

        """
        The next step is to apply contour detection to extract any motion regions:
        """
        # find contours in the thresholded image and initialize the minimum and maximum
        # bounding box regions for motion
        """
        perform contour detection on our thresh image:
        """
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        """
        We then initialize two sets of bookkeeping variables to keep track of the 
        location where any motion is contained. These variables will form the “bounding box” 
        which will tell us the location of where the motion is taking place
        """
        (minX, minY) = (np.inf, np.inf)
        (maxX, maxY) = (-np.inf, -np.inf)

        """
        The final step is to populate these variables (provided motion exists in the frame, 
        of course):
        """
        # if no contours were found, return None
        """
        we make a check to see if our contours list is empty:
        """
        if len(cnts) == 0:
            """
            If that’s the case, then there was no motion found in the frame and we can 
            safely ignore it:
            """
            return None

        """
        Otherwise, motion does exist in the frame so we need to start looping over the 
        contours:
        """
        # otherwise, loop the contours
        for c in cnts:
            # compute the bounding box of the contour and use it to update the minimum and 
            # maximum bounding box regions
            """
            For each contour we compute the bounding box and then update our bookkeeping 
            variables (Lines 47-53), finding the minimum and maximum (x, y)-coordinates that 
            all motion has taken place it:
            """
            (x, y, w, h) = cv2.boundingRect(c)
            (minX, minY) = (min(minX, x), min(minY, y))
            (maxX, maxY) = (max(maxX, x + w), max(maxY, y + h))

        # otherwise, return a tuple of the threshold image along with bounding box
        return (thresh, (minX, minY, maxX, maxY))

