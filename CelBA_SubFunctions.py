# Importing required functions for first part
import cv2 as cv
import glob
import numpy as np
import copy
import math
import os
import pandas as pd

from numpy.ma.core import transpose
from skimage.morphology import skeletonize
import scipy.interpolate

# ---------------------------------------- functions to check size, overlaps, begin making contours --------------------

# Resolution that all the images are resized to.
# Jack would suggest setting this to the resolution of the original images.
# Jack included for testing how the parameters changed with different resolutions
resolution = 1024

# Folder created for where the intermediate files and results should be saved from CelBA
# They will be saved in this folder, with the name of the folder the image sequence was originally in when selected by the GUI
outputdirectory = "/Users/caitlincutts/Documents/Cam/4th Yr/Project/CelBAfinal_zipup/Processed/"


def find_first_frame_contours(folder, size, centroids=False):
    '''
    :param folder: absolute path to folder containing frames as a string (no / on the end)
    :param size: threshold area of contour to be considered a worm
    :return: list containg images as np.arrays - each image is the contour of a single worm in the first frame, wormcount
    :return: wormcount
    :return: areas of each of the worms in the first frame
    :return: centroid location of each worm in the first frame (if input False this is none to save time)
    '''

    folderpaths = str(folder) + "/*.jpg"
    # foldername = os.path.basename(folder)
    # overalloutput = outputdirectory + foldername

    # Loads images and converts to binary, performs thresholding, creates contours
    imagename = sorted(glob.glob(folderpaths))[0]
    imagebig = cv.imread(imagename)
    image = cv.resize(imagebig, (resolution, resolution))
    imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Iterates through contours generated above, selecting them if they are larger than our size requirement.
    large_contour = []
    centroid_list = []
    this_frame_areas = []
    # i=0
    for contour in contours:
        area = cv.contourArea(contour)
        # print(i, area)
        # i += 1
        if area > int(size):
            # print("big enough")
            large_contour.append(contour)
            this_frame_areas = cv.contourArea(contour)

    wormcount = len(large_contour)

    # Iterates through worms, draws them on a blank image so we can see if our chosen size value is correct. This really doesn't need much tuning, unless magnification dramatically changed.
    # Instead of being saved they are returned in the function
    images = []
    for worms in range(0, wormcount):
        # print(f"worm {worms} started!")
        blank = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        # Draws on blank background filled contour of our largest,  contours in green
        images.append(cv.drawContours(blank, [large_contour[worms]], -1, (0, 255, 0), thickness=cv.FILLED))

        if centroids:
            M = cv.moments(large_contour[worms])
            if M["m00"] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            centroidwormi = [[cx, cy]]
            centroid_list.append(centroidwormi)
        else:
            centroid_list = None

    print(f"size check for size = {size} done")
    return images, wormcount, this_frame_areas, centroid_list

def draw_true_sections(bool_list, canvas, framecount):
    """Draw rectangles from a boolean list with optional editing."""
    canvas.delete("all")  # Clear canvas before redrawing
    rect_y1, rect_y2 = 8, 18
    scale_factor = 500 / framecount
    start_index = None
    rectangles = []  # Store drawn rectangles

    for i, val in enumerate(bool_list):
        if val and start_index is None:
            start_index = i  # Start of a True section
        elif not val and start_index is not None:
            # Draw rectangle
            x1 = start_index * scale_factor
            x2 = i * scale_factor
            rect_id = canvas.create_rectangle(x1, rect_y1, x2, rect_y2, fill="red", outline="black", tags="static")
            # print(f"Drawn rectangle at ({x1}, {x2}) for indices {start_index} to {i}")
            start_index = None

    # Handle last rectangle if still open
    if start_index is not None:
        x1 = start_index * scale_factor
        x2 = len(bool_list) * scale_factor
        rect_id = canvas.create_rectangle(x1, rect_y1, x2, rect_y2, fill="red", outline="black", tags="static")
        # print(f"Final rectangle at ({x1}, {x2}) for indices {start_index} to {len(bool_list)}")

    return rectangles  # Return list of drawn rectangles

class EditableRectangles:
    def __init__(self, canvas, bool_list, framecount):
        self.canvas = canvas

        self.bool_list = bool_list
        self.framecount = framecount
        self.scale_factor = 500 / framecount
        self.rectangles = []

        self.selected_rect = None  # Track selected rectangle for resizing
        self.start_x = None  # Track original mouse click position
        self.drawing_new = False  # Flag for new rectangle drawing
        self.new_start_idx = None  # Track where new rectangle starts

        self.draw_rectangles()

        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def draw_rectangles(self):
        """Draw rectangles from the boolean list"""
        self.canvas.delete("all")  # Clear canvas before redrawing
        self.rectangles = []

        rect_y1, rect_y2 = 6, 28
        start_index = None

        for i, val in enumerate(self.bool_list):
            if val and start_index is None:
                start_index = i  # Start of a True section
            elif not val and start_index is not None:
                # Draw rectangle and store its reference
                x1 = start_index * self.scale_factor
                x2 = i * self.scale_factor
                rect_id = self.canvas.create_rectangle(x1, rect_y1, x2, rect_y2, fill="blue", outline="black")
                self.rectangles.append((rect_id, start_index, i))
                start_index = None

        if start_index is not None:
            x1 = start_index * self.scale_factor
            x2 = len(self.bool_list) * self.scale_factor
            rect_id = self.canvas.create_rectangle(x1, rect_y1, x2, rect_y2, fill="blue", outline="black")
            self.rectangles.append((rect_id, start_index, len(self.bool_list)))

    def on_click(self, event):
        """Detects clicks to either resize or create new rectangles"""
        for rect_id, start_idx, end_idx in self.rectangles:
            x1, _, x2, _ = self.canvas.coords(rect_id)
            if x1 <= event.x <= x2:  # Click inside a rectangle (resize mode)
                self.selected_rect = (rect_id, start_idx, end_idx)
                self.start_x = event.x
                return

        # If no existing rectangle is clicked, start drawing a new one
        self.drawing_new = True
        self.new_start_idx = int(event.x / self.scale_factor)

    def on_drag(self, event):
        """Handles both resizing existing rectangles and drawing new ones"""
        if self.selected_rect:
            rect_id, start_idx, end_idx = self.selected_rect
            x1, _, x2, _ = self.canvas.coords(rect_id)

            # Update rectangle width
            new_x2 = max(x1 + 5, event.x)
            self.canvas.coords(rect_id, x1, 6, new_x2, 28)

            # Update the bool_list
            new_end_idx = min(int(new_x2 / self.scale_factor), len(self.bool_list))
            for i in range(start_idx, end_idx):
                self.bool_list[i] = False  # Clear old range
            for i in range(start_idx, new_end_idx):
                self.bool_list[i] = True  # Set new range

        elif self.drawing_new:
            # Preview rectangle being drawn
            new_x2 = min(event.x, self.canvas.winfo_width())
            self.canvas.delete("temp_rect")
            self.canvas.create_rectangle(
                self.new_start_idx * self.scale_factor, 6, new_x2, 28, fill="lightblue", outline="black",
                tags="temp_rect"
            )

    def on_release(self, event):
        """Finalizes rectangle movement or creation"""
        if self.selected_rect:
            self.selected_rect = None  # End resizing

        elif self.drawing_new:
            self.drawing_new = False
            new_end_idx = int(event.x / self.scale_factor)
            if new_end_idx > self.new_start_idx:  # Ensure it's a valid rectangle
                for i in range(self.new_start_idx, new_end_idx):
                    self.bool_list[i] = True  # Mark new section as True
            self.draw_rectangles()  # Redraw with new rectangle

class look_for_overlaps():
    def __init__(self, folder, size, threshold, wormcount):
        '''
        :param folder: folder containing images you want to look through for overlaps
        :param size: threshold area of contour to be considered a worm
        :param threshold: threshold number of frames can be on their own between overlap chunks
        :.traced_images: series of images of detected worms bound by rectangles
        :.overlap_frames: list, True for any frames where there are fewer rectangles than wormcount
        :.overlap_chunk: list, True for all overlapping frames and any small chunks between (<5 frames)
        '''

        folderpaths = str(folder) + "/*.jpg"
        framepaths = sorted(glob.glob(folderpaths))

        self.traced_images = []
        self.overlap_frames = np.full((len(framepaths)), False)
        for frameindex, frame in enumerate(framepaths):
            # Read in image
            imagebig = cv.imread(frame)
            image = cv.resize(imagebig, (resolution, resolution))
            imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            ret, thresh = cv.threshold(imgray, 127, 255, 0)
            wormcontours, wormhierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            this_frame_rectangles = []

            drawn_contours = 0
            for c in wormcontours:

                # Check if the contour area is larger than a threshold.
                if cv.contourArea(c) > size - size*0.5:
                    # Get the bounding rectangle coordinates.
                    (x, y, w, h) = cv.boundingRect(c)
                    this_frame_rectangles.append([x, y, w, h])
                    centre = np.array([int(x+w/2), int(y+h/2)])

                    M = cv.moments(c)
                    if M["m00"] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                    centroidwormi = [cx, cy]

                    image = cv.circle(image, centroidwormi, radius=0, color=(255, 0, 0), thickness=4)
                    image = cv.circle(image, centre, radius=0, color=(0, 255, 0), thickness=4)


                    # Draw a blue rectangle around the contour.
                    cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    drawn_contours += 1
            if drawn_contours < wormcount:
                self.overlap_frames[frameindex] = True

            self.traced_images.append(image)

        self.overlap_chunk = copy.deepcopy(self.overlap_frames)
        overlap_chunk_started = False
        length_False_chunk = 0

        for i, value in enumerate(self.overlap_frames):

            # if there is a True we have entered a chunk of overlaps
            if value:
                # if there is another overlap and we are still in an overlap chunk then we can set all the intermed frames to be blacked out
                if overlap_chunk_started:
                    for intermediate_frames in range(1, length_False_chunk + 1):
                        self.overlap_chunk[i-intermediate_frames] = True
                overlap_chunk_started = True

            # if we have had some overlaps but now there is no overlap, measure number of frames since the last overlap
            if value == False and overlap_chunk_started:
                length_False_chunk +=1

                # if there have been 5 frames since the last overlap we can consider the overlap chunk over, reset count
                if length_False_chunk == threshold:
                    overlap_chunk_started = False
                    length_False_chunk = 0

def find_ends(linetrans):
    ends = []
    #Creates empty vector to store the ends of the worm
    for irow in range(0,len(linetrans)):
        samey = linetrans[irow][0]
        samex = linetrans[irow][1]
        plus1y = samey + 1
        minus1y = samey - 1
        plus1x = samex + 1
        minus1x = samex - 1

        #Possible next positions for the skeleton
        a = np.array([samey,plus1x])
        b = np.array([samey,minus1x])
        c = np.array([plus1y,samex])
        d = np.array([plus1y,plus1x])
        e = np.array([plus1y,minus1x])
        f = np.array([minus1y,samex])
        g = np.array([minus1y,plus1x])
        h = np.array([minus1y,minus1x])

        #Counts how many of these next positions are filled
        count = 0
        for istep in (a,b,c,d,e,f,g,h):
            if np.isin(linetrans,istep).all(1).any() == True:
                count = count + 1
        #If only one of these next positions is filled, then we are at the end of the worm
        if count < 2:
            ends.append([samey,samex])


    return(ends)

def ordered_line(lastend, ends, linetrans):
    if len(ends) > 3:
        # A worm should not have this many ends, only occurs if covered in bacteria or something, this solution gives a result starting from the first end.
        numends = 4

        # Extracts start coordinate from the first end
        if len(lastend) != 0:
            ygap0 = abs(ends[0][0] - lastend[0])
            xgap0 = abs(ends[0][1] - lastend[1])
            ygap1 = abs(ends[1][0] - lastend[0])
            xgap1 = abs(ends[1][1] - lastend[1])
            ygap2 = abs(ends[2][0] - lastend[0])
            xgap2 = abs(ends[2][1] - lastend[1])
            ygap3 = abs(ends[3][0] - lastend[0])
            xgap3 = abs(ends[3][1] - lastend[1])

            gap0 = ygap0 + xgap0
            gap1 = ygap1 + xgap1
            gap2 = ygap2 + xgap2
            gap3 = ygap3 + xgap3

            if gap0 < gap1 and gap0 < gap2 and gap0 < gap3:
                starty = ends[0][0]
                startx = ends[0][1]

            if gap1 < gap0 and gap1 < gap2 and gap1 < gap3:
                starty = ends[1][0]
                startx = ends[1][1]

            if gap2 < gap0 and gap2 < gap1 and gap2 < gap3:
                starty = ends[2][0]
                startx = ends[2][1]

            if gap3 < gap0 and gap3 < gap1 and gap3 < gap2:
                starty = ends[3][0]
                startx = ends[3][1]

            else:
                starty = ends[0][0]
                startx = ends[0][1]

        else:
            starty = ends[0][0]
            startx = ends[0][1]

        startstep = np.array([starty, startx])

        lastend = startstep

        # Creates array, and then dataframe containing all the possible steps that could have been taken, updates later
        donesteps = [startstep]

        # Creates array, and then dataframe containing all the steps that were taken, updates later
        finalsteps =[list(startstep)]

        # For every pixel in the skeleton, computes potential steps
        for irow in range(0, len(linetrans)):
            plus1y = starty + 1
            minus1y = starty - 1
            plus1x = startx + 1
            minus1x = startx - 1

            # Defines potential steps, order of abc etc defines whether we see more diagnoalised or straight pixel jumps.
            # This order uses diagonal jumps first
            h = np.array([starty, plus1x])
            e = np.array([starty, minus1x])
            g = np.array([plus1y, startx])
            a = np.array([plus1y, plus1x])
            b = np.array([plus1y, minus1x])
            f = np.array([minus1y, startx])
            c = np.array([minus1y, plus1x])
            d = np.array([minus1y, minus1x])

            # Count variable to ensure only one step made per pixel
            count = 0
            # For potential steps, checks if it is in the donetable, ensuring no backwards steps
            # If the step is not in the done table, adds it to the finaltable, and updates start position, increases count to 1, move on to next position.
            for istep in (a, b, c, d, e, f, g, h):
                if (linetrans == istep).all(1).any() == True:
                    if (donesteps == istep).all(1).any() == False:
                        donesteps.append(istep)
                        if count == 0:
                            finalsteps.append(list(istep))
                            starty = istep[0]
                            startx = istep[1]
                            count = 1

    if len(ends) == 3:
        # Again, worm should not really have 3 ends. Chooses to start skeleton from the starting position in the previous frame.
        numends = 3

        # Extracts start coordinate from the first end
        if len(lastend) != 0:
            ygap0 = abs(ends[0][0] - lastend[0])
            xgap0 = abs(ends[0][1] - lastend[1])
            ygap1 = abs(ends[1][0] - lastend[0])
            xgap1 = abs(ends[1][1] - lastend[1])
            ygap2 = abs(ends[2][0] - lastend[0])
            xgap2 = abs(ends[2][1] - lastend[1])

            gap0 = ygap0 + xgap0
            gap1 = ygap1 + xgap1
            gap2 = ygap2 + xgap2

            if gap0 < gap1 and gap0 < gap2:
                starty = ends[0][0]
                startx = ends[0][1]

            if gap1 < gap0 and gap1 < gap2:
                starty = ends[1][0]
                startx = ends[1][1]

            if gap2 < gap0 and gap2 < gap1:
                starty = ends[2][0]
                startx = ends[2][1]
            else:
                starty = ends[0][0]
                startx = ends[0][1]

        else:
            starty = ends[0][0]
            startx = ends[0][1]

        startstep = np.array([starty, startx])

        lastend = startstep

        # Creates array, and then dataframe containing all the possible steps that could have been taken, updates later
        donesteps = [startstep]

        # Creates array, and then dataframe containing all the steps that were taken, updates later
        finalsteps = [list(startstep)]

        # For every pixel in the skeleton, computes potential steps
        for irow in range(0, len(linetrans)):
            plus1y = starty + 1
            minus1y = starty - 1
            plus1x = startx + 1
            minus1x = startx - 1

            # Defines potential steps, order of abc etc defines whether we see more diagnoalised or straight pixel jumps.
            # This order uses diagonal jumps first
            h = np.array([starty, plus1x])
            e = np.array([starty, minus1x])
            g = np.array([plus1y, startx])
            a = np.array([plus1y, plus1x])
            b = np.array([plus1y, minus1x])
            f = np.array([minus1y, startx])
            c = np.array([minus1y, plus1x])
            d = np.array([minus1y, minus1x])

            # Count variable to ensure only one step made per pixel
            count = 0
            # For potential steps, checks if it is in the donetable, ensuring no backwards steps
            # If the step is not in the done table, adds it to the finaltable, and updates start position, increases count to 1, move on to next position.
            for istep in (a, b, c, d, e, f, g, h):
                if (linetrans == istep).all(1).any() == True:
                    if (donesteps == istep).all(1).any() == False:
                        donesteps.append(istep)
                        if count == 0:
                            finalsteps.append(list(istep))
                            starty = istep[0]
                            startx = istep[1]
                            count = 1

    if len(ends) == 2:
        # This should be a linear worm
        numends = 2
        # Extracts start coordinate from the first end
        if len(lastend) != 0:
            ygap0 = abs(ends[0][0] - lastend[0])
            xgap0 = abs(ends[0][1] - lastend[1])
            ygap1 = abs(ends[1][0] - lastend[0])
            xgap1 = abs(ends[1][1] - lastend[1])
            gap0 = ygap0 + xgap0
            gap1 = ygap1 + xgap1

            if gap0 < gap1:
                starty = ends[0][0]
                startx = ends[0][1]

            else:
                starty = ends[1][0]
                startx = ends[1][1]

        else:
            starty = ends[0][0]
            startx = ends[0][1]

        startstep = np.array([starty, startx])

        lastend = startstep

        # Creates array, and then dataframe containing all the possible steps that could have been taken, updates later
        donesteps = [startstep]

        # Creates array, and then dataframe containing all the steps that were taken, updates later
        finalsteps = [list(startstep)]

        # For every pixel in the skeleton, computes potential steps
        for irow in range(0, len(linetrans)):
            plus1y = starty + 1
            minus1y = starty - 1
            plus1x = startx + 1
            minus1x = startx - 1

            # Defines potential steps, order of abc etc defines whether we see more diagnoalised or straight pixel jumps.
            # This order uses diagonal jumps first
            h = np.array([starty, plus1x])
            e = np.array([starty, minus1x])
            g = np.array([plus1y, startx])
            a = np.array([plus1y, plus1x])
            b = np.array([plus1y, minus1x])
            f = np.array([minus1y, startx])
            c = np.array([minus1y, plus1x])
            d = np.array([minus1y, minus1x])

            # Count variable to ensure only one step made per pixel
            count = 0
            # For potential steps, checks if it is in the donetable, ensuring no backwards steps
            # If the step is not in the done table, adds it to the finaltable, and updates start position, increases count to 1, move on to next position.
            for istep in (a, b, c, d, e, f, g, h):
                if (linetrans == istep).all(1).any() == True:
                    if (donesteps == istep).all(1).any() == False:
                        donesteps.append(istep)
                        if count == 0:
                            finalsteps.append(list(istep))
                            starty = istep[0]
                            startx = istep[1]
                            count = 1
    if len(ends) == 1:
        numends = 1
        # This is for a worm in a "6" shape, from each frame to the next, it should go around the 6 in the same way.
        starty = ends[0][0]
        startx = ends[0][1]

        startstep = np.array([starty, startx])

        lastend = startstep

        # Creates array, and then dataframe containing all the possible steps that could have been taken, updates later
        donesteps = [startstep]

        # Creates array, and then dataframe containing all the steps that were taken, updates later
        finalsteps = [list(startstep)]

        # For every pixel in the skeleton, computes potential steps
        for irow in range(0, len(linetrans)):
            plus1y = starty + 1
            minus1y = starty - 1
            plus1x = startx + 1
            minus1x = startx - 1

            # Defines potential steps, order of abc etc defines whether we see more diagnoalised or straight pixel jumps.
            # This order uses diagonal jumps first
            h = np.array([starty, plus1x])
            e = np.array([starty, minus1x])
            g = np.array([plus1y, startx])
            a = np.array([plus1y, plus1x])
            b = np.array([plus1y, minus1x])
            f = np.array([minus1y, startx])
            c = np.array([minus1y, plus1x])
            d = np.array([minus1y, minus1x])

            # Count variable to ensure only one step made per pixel
            count = 0
            # For potential steps, checks if it is in the donetable, ensuring no backwards steps
            # If the step is not in the done table, adds it to the finaltable, and updates start position, increases count to 1, move on to next position.
            for istep in (a, b, c, d, e, f, g, h):
                if (linetrans == istep).all(1).any() == True:
                    if (donesteps == istep).all(1).any() == False:
                        donesteps.append(istep)
                        if count == 0:
                            finalsteps.append(list(istep))
                            starty = istep[0]
                            startx = istep[1]
                            count = 1
    if len(ends) == 0:
        numends = 0
        # This is a curled worm, again from one frame to the next, it should go the same way around the worm
        starty = linetrans[0][0]
        startx = linetrans[0][1]

        startstep = np.array([starty, startx])

        lastend = startstep

        # Creates array, and then dataframe containing all the possible steps that could have been taken, updates later
        donesteps = [startstep]

        # Creates array, and then dataframe containing all the steps that were taken, updates later
        finalsteps = [list(startstep)]

        # For every pixel in the skeleton, computes potential steps
        for irow in range(0, len(linetrans)):
            plus1y = starty + 1
            minus1y = starty - 1
            plus1x = startx + 1
            minus1x = startx - 1

            # Defines potential steps, order of abc etc defines whether we see more diagnoalised or straight pixel jumps.
            # This order uses diagonal jumps first
            h = np.array([starty, plus1x])
            e = np.array([starty, minus1x])
            g = np.array([plus1y, startx])
            a = np.array([plus1y, plus1x])
            b = np.array([plus1y, minus1x])
            f = np.array([minus1y, startx])
            c = np.array([minus1y, plus1x])
            d = np.array([minus1y, minus1x])

            # Count variable to ensure only one step made per pixel
            count = 0
            # For potential steps, checks if it is in the donetable, ensuring no backwards steps
            # If the step is not in the done table, adds it to the finaltable, and updates start position, increases count to 1, move on to next position.
            for istep in (a, b, c, d, e, f, g, h):
                if (linetrans == istep).all(1).any() == True:
                    if (donesteps == istep).all(1).any() == False:
                        donesteps.append(istep)
                        if count == 0:
                            finalsteps.append(list(istep))
                            starty = istep[0]
                            startx = istep[1]
                            count = 1
    return (lastend, finalsteps)

# Absorbed makeskeletons, overlayskeletonfullimage and start contours into one function
def worm_tracking(folder, size, bool_vector, contourstatus):
    '''
    :param folder: absolute path to folder containing frames as a string (no / on the end)
    :param size: threshold area of contour to be considered a worm
    :param wormcount: wormcount found from previous function
    :param contourstatus: update GUI progress bar function
    :return: Wormcount
    :return: centroid_list also same dimensions except each value is a coordinate of the centroid of that worm in that frame
    :return: GUI_images with the centroid path, current centroid and skeleton of the worm on
    :return: framecount, table of frames, each frame containts a variable for each worm, each of these is a list of coordinates of the skeleton
    '''

    first_frame_images, wormcount, prev_frame_areas, prev_frame_centroids = find_first_frame_contours(folder, size,True)

    centroid_list = []
    previous_wormi_area = []
    numendstable = []
    one_per_pic = []
    for i in range(wormcount):
        previous_wormi_area.append([])
        one_per_pic.append([])
        numendstable.append([])
    foldername = str(folder) + "/*.jpg"
    framepaths = sorted(glob.glob(foldername))
    framecount = len(framepaths)
    table = []


    lastend = []
    GUI_images = []
    worms_in_overlap = [False]*wormcount

    if len(bool_vector) < framecount:
        bool_vector = [False] * framecount

    progress = 0

    for i in prev_frame_centroids:
        centroid_list.append(i)

    # Iterate through frames in folder
    for framenumber, frame in enumerate(framepaths):

        # Read in frame, generate contours
        imagebig = cv.imread(frame)
        image = cv.resize(imagebig, (resolution, resolution))
        imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(imgray, 127, 255, 0)
        wormcontours, wormhierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # worms are contours above a certain area, this is a size filter

        # first initialise some variables
        large_contours = []
        this_frame_centroids = []
        this_frame_areas = []
        this_frame_skeletons = []

        # loop contours in this frame through the size filter
        for wormcontour in wormcontours:
            area = cv.contourArea(wormcontour)
            if area > size - size * 0.5:  # -50 here to ensure that we don't lose any worms that are just smaller than our decided size value in certain frames
                large_contours.append(wormcontour)
                M = cv.moments(wormcontour)
                if M["m00"] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                this_frame_centroids.append([cx, cy])
                this_frame_areas.append(area)

        # If this is the first non-overlap frame of a chunk centroid_list[worm][-1] is None so we need to find the new centroid location
        if bool_vector[framenumber] == False and bool_vector[framenumber - 1] == True:
            new_centroids = copy.deepcopy(this_frame_centroids)
            non_overlap_centroids = []
            for worm in range(wormcount):
                non_overlap_centroids.append(centroid_list[worm][-1])

            for centroid in non_overlap_centroids:
                if centroid is not None:
                    distances = [math.dist(centroid, new_centroid) for new_centroid in new_centroids]
                    smallest_distance_index = np.argmin(distances)
                    new_centroids.pop(smallest_distance_index)
                    # new_centroids now only contains centroids which were not assigned last frame

        # If we have entered an overlap chunk
        # print(bool_vector[framenumber], np.isin(worms_in_overlap, False).all())
        if bool_vector[framenumber] and np.isin(worms_in_overlap, False).all():
            # print("entering overlap chunk")
            overlap_centroids = []
            for worm in range(wormcount):
                # print(worm)

                # Retrieve previous centroid
                previous_centroid = centroid_list[worm][-1]
                distances = [math.dist(previous_centroid, centroid) for centroid in this_frame_centroids]

                # Get the closest centroid index then remove
                smallest_distance_index = np.argmin(distances)
                # Retrieve corresponding centroid
                wormicentroid = this_frame_centroids[smallest_distance_index]

                # wormicontour = large_contours[smallest_distance_index]

                # if this wormcontour has already been assigned then we have found the overlap
                if len(overlap_centroids) > 0:
                    worms_in_overlap = np.isin(overlap_centroids, wormicentroid).all(1)
                    if worms_in_overlap.any():
                        worms_in_overlap = np.ndarray.tolist(worms_in_overlap)
                        worms_in_overlap.append(True)
                        while len(worms_in_overlap) < wormcount:
                            worms_in_overlap.append(False)
                        # print(worms_in_overlap, len(worms_in_overlap),"exiting first overlap frame")
                        break
                    else:
                        overlap_centroids.append(wormicentroid)
                else:
                    overlap_centroids.append(wormicentroid)

        # Business as usual, there is no overlap or this worm is not involved in a current overlap
        for worm in range(wormcount):
            if bool_vector[framenumber] == False or worms_in_overlap[worm] == False:
                # print("worm ID", worm, '=', worms_in_overlap[worm])
                # print(f"framenumber: {framenumber}, worm: {worm}")
                # Initialize blank image
                blank = np.zeros((resolution, resolution, 3), dtype=np.uint8)

                # Retrieve previous centroid
                previous_centroid = centroid_list[worm][-1]
                # print(previous_centroid)

                if previous_centroid is not None:
                    distances = [math.dist(previous_centroid, centroid) for centroid in this_frame_centroids]

                    # Get the closest centroid index
                    smallest_distance_index = np.argmin(distances)

                    # Retrieve corresponding contour and centroid
                    wormicentroid = this_frame_centroids[smallest_distance_index]
                    wormicontour = large_contours[smallest_distance_index]
                    wormiarea = this_frame_areas[smallest_distance_index]

                else:
                    wormicentroid = new_centroids[0]
                    new_centroids.pop(0)
                    index =this_frame_centroids.index(wormicentroid)
                    wormicontour = large_contours[index]
                    wormiarea = this_frame_areas[index]

                # Store updated centroid
                centroid_list[worm].append(wormicentroid)

                # Identify smaller contours (potential noise) to be filled in black
                black_contours = [contour for contour in wormcontours if cv.contourArea(contour) < wormiarea]

                # Draw contours on blank image
                cv.drawContours(blank, [wormicontour], -1, (0, 255, 0), thickness=cv.FILLED)
                cv.drawContours(blank, black_contours, -1, (0, 0, 0), thickness=cv.FILLED)

                one_per_pic[worm].append(blank)  # This is for activity parameter calculation

                # Convert to grayscale and skeletonize
                gray_blank = cv.cvtColor(blank, cv.COLOR_BGR2GRAY)
                binary_image = (gray_blank > 127).astype(np.uint8)
                thinned = skeletonize(binary_image)

                # Extract and process skeleton points
                line = np.nonzero(thinned)
                linetrans = np.transpose(line)
                thispicends = find_ends(linetrans)
                lastend, thispictable = ordered_line(lastend, thispicends, linetrans)
                # Store results
                this_frame_skeletons.append(list(thispictable))
                numendstable[worm].append([thispicends])
                # print(worm, framenumber, thispicends)

            else:
                centroid_list[worm].append(None)
                this_frame_skeletons.append(None)
                numendstable[worm].append(None)
                one_per_pic[worm].append(None)

        # In each frame we want to draw the skeletons onto the image to put into the GUI
        this_frame_images = []
        for worm in range(wormcount):
            image = cv.resize(imagebig, (resolution, resolution))
            # print(f"framenumber: {framenumber}, worm: {worm}")

            if bool_vector[framenumber] == False or worms_in_overlap[worm] == False:
                for pixels in this_frame_skeletons[worm]:
                    [y, x] = pixels
                    cv.circle(image, [x, y], radius=0, color=(255, 0, 0), thickness=2)

                for pixels in centroid_list[worm]:
                    # print(pixels)
                    cv.circle(image, pixels, radius=0, color=(0, 255, 0), thickness=2)

                point = centroid_list[worm][-1]
                # print(point)
                cv.circle(image, point, radius=0, color=(0, 0, 255), thickness=2)

            this_frame_images.append(image)

            # Feedback function to the GUI, to update the progress bar
            contourstatus(progress, 1, len(sorted(glob.glob(foldername)) * wormcount))
            progress = progress + 1
            # print(f"framenumber: {framenumber}, worm: {worm}")

        table.append(this_frame_skeletons)
        GUI_images.append(this_frame_images)


        # If this is the last overlapping frame in the chunk, reset worms_in_overlap
        if framenumber < framecount - 1:
            if bool_vector[framenumber +1 ] == False and bool_vector[framenumber] == True:
                worms_in_overlap = [False] * wormcount

    # print(centroid_list)
    return framecount, wormcount, centroid_list, GUI_images, table, numendstable, one_per_pic

def list_switch(list, wormA, wormB, start, end):
    '''
    :param list: List of parameters through the frames for all worms
    :param wormA: Worms which need switching
    :param wormB: "
    :param start: Index of the first frame of the chunk being switched
    :param end: Index of the last frame of the chunk being switched
    :return: list with the chunks of appropriate columns switched
    '''


    listA = list[wormA]
    listB = list[wormB]

    # print("List A before swap:", listA[start:end + 1])
    # print("List B before swap:", listB[start:end + 1])

    # Deepcopy the slice from listA
    store = copy.deepcopy(listA[start:end + 1])

    # Swap the slices
    listA[start:end + 1] = listB[start:end + 1]
    listB[start:end + 1] = store

    # print("List A after swap:", listA[start:end + 1])
    # print("List B after swap:", listB[start:end + 1])

    return list

# ----------------------------------------Calculating Parameters--------------------------------------------------------

def kappa(wormcount, framecount, skeletons):
    #This is using numerical integration, so we can get +ve and -ve curvature, other methods only give one sign.

    totalcurvature = []
    bodylengths = []
    #Iterates through all worms
    for worms in range(wormcount):
        prev_body_length = 0
        thiswormcurvature = []
        bodylengths.append([])
        #Iterates through all frames
        for iframes in range(framecount):
            skel_coords = np.array(skeletons[iframes][worms])

            if skel_coords.ndim == 0:
                print ("True")
                thiswormcurvature.append(None)

            else:

                framecurvature = [] # contains the curvature for each of the segments along the body

                # Compute segment-wise distances
                dists = np.sqrt(np.sum(np.diff(skel_coords, axis=0) ** 2, axis=1))
                body_length = np.sum(dists)
                distances = np.cumsum(dists)
                bodylengths[worms].append(body_length)

                if body_length < 0.5 * prev_body_length:
                    thiswormcurvature.append(None)

                else:
                    prev_body_length = body_length
                    distances = np.insert(distances, 0, 0)  # Insert 0 at the start
                    # Generate 12 equally spaced distances along the total path
                    new_distances = np.linspace(0, distances[-1], 12)
                    # Interpolate new x and y coordinates
                    x_interp = scipy.interpolate.interp1d(distances, skel_coords[:, 0], kind='linear')
                    y_interp = scipy.interpolate.interp1d(distances, skel_coords[:, 1], kind='linear')
                    new_x = x_interp(new_distances)
                    new_y = y_interp(new_distances)
                    segmented_skeleton = np.column_stack((new_x, new_y))

                    for i in range(0,10):
                        #Based of formula for numerical integration, calculation of curvature from three points.
                        xs0 = segmented_skeleton[i, 0]
                        xs1 = segmented_skeleton[i+1, 0]
                        xs2 = segmented_skeleton[i+2, 0]
                        xs0dash = (xs1-xs0)/(1/12)
                        xs1dash = (xs2-xs1)/(1/12)
                        xs0doubledash = (xs1dash-xs0dash)/(1/12)

                        ys0 = segmented_skeleton[i, 1]
                        ys1 = segmented_skeleton[i+1, 1]
                        ys2 = segmented_skeleton[i+2, 1]
                        ys0dash = (ys1-ys0)/(1/12)
                        ys1dash = (ys2-ys1)/(1/12)
                        ys0doubledash = (ys1dash-ys0dash)/(1/12)

                        #kappa is negative for clockwise, positive for anti?
                        kappa = round(((xs0dash*ys0doubledash)-(xs0doubledash*ys0dash))/((((xs0dash)**2)+((ys0dash)**2))**(3/2)),2)
                        framecurvature.append(kappa)

                    thiswormcurvature.append(framecurvature)

        totalcurvature.append(thiswormcurvature)

    return totalcurvature, bodylengths

def count_curling(wormcount, framecount, numendstable, bodylengths):

    thiswormcurl = []
    for i in range(wormcount):
        thiswormcurl.append([])

    for worm in range(wormcount):
        this_worm_bodylengths = np.array(bodylengths[worm])
        this_worm_ave_length = np.mean(this_worm_bodylengths)
        # print(this_worm_ave_length)

        thiswormends = numendstable[worm]

        for frame in range(framecount):
            thisframeends = thiswormends[frame]

            if thisframeends is None:
                thiswormcurl[worm].append(None)

            else:
                thisframeends = thisframeends[0]
                numends = len(thisframeends)
                if numends < 2:
                    thiswormcurl[worm].append(1)
                    # print(numends, "option A")

                elif numends == 2 and math.dist(thisframeends[0], thisframeends[1]) < (1/3)*this_worm_ave_length:
                    thiswormcurl[worm].append(1)
                    # print(numends, "option B", (1/3)*this_worm_ave_length, math.dist(thisframeends[0], thisframeends[1]))

                else:
                    thiswormcurl[worm].append(0)


    print("curling calculated!")
    return thiswormcurl

def calc_curvature_params(wormcount, framecount, curvature):
    totalasymmetry = []
    totalbend = []
    totalstretch = []
    for worms in range(0, wormcount):

        wormasymmetry = []
        wormbend = []
        wormstretch = []

        table = curvature[worms]

        for iframes in range(framecount):


            if table[iframes] is None:
                wormasymmetry.append(None)
                wormbend.append(None)
                wormstretch.append(None)

            else:
                # Asymmetry
                curvatures = np.array(table[iframes])
                thisframewormasymmetry = abs(round(sum(curvatures), 2))
                if thisframewormasymmetry > 2:
                    thisframewormasymmetry = 2
                wormasymmetry.append(thisframewormasymmetry)

                # Bend

                thisframewormbend = round(sum(abs(curvatures)), 2)
                if thisframewormbend > 4:
                    thisframewormbend = 4
                wormbend.append(thisframewormbend)

                # Stretch
                absolute = abs(curvatures)
                max = np.max(absolute)
                min = np.min(absolute)
                stretch = round(max - min, 2)
                if stretch > 2:
                    stretch = 2
                wormstretch.append(stretch)

        totalasymmetry.append(wormasymmetry)
        totalbend.append(wormbend)
        totalstretch.append(wormstretch)

    return totalasymmetry, totalbend, totalstretch

# This function is used in activity to calculate the ranges where activity can be caluclated (not overlapping)
def find_missing_ranges(list, framecount):
    full_range = set(range(framecount + 1))
    given_set = set(list)
    missing = sorted(full_range - given_set)

    chunks = []
    start = missing[0]

    for i in range(1, len(missing)):
        if missing[i] != missing[i - 1] + 1:
            chunks.append([start, missing[i - 1]])
            start = missing[i]

    chunks.append([start, missing[-1]])
    return chunks

def activity(wormcount, framecount, activity_pics):

    # find the average area of each worm
    ave_worm_areas = []
    none_frames = [] # List of the frame numbers where there was no overlap so no image recorded
    for worms in range(0, wormcount):
        none_frames.append([])
        this_worm_total_frames = activity_pics[worms]
        sum_this_worm_contour_areas = 0

        for images in range(0, framecount):
            newpic = this_worm_total_frames[images]
            if newpic is None:
                none_frames[worms].append(images)
            else:
                #print(newpic)
                imgray = cv.cvtColor(newpic, cv.COLOR_BGR2GRAY)
                # Looks at image contrast, gives threshold value for contour
                ret, thresh = cv.threshold(imgray, 127, 255, 0)
                # Lists contours
                contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

                cv.drawContours(imgray, contours, -1, (0,255,0), 2)


                contour_area = cv.contourArea(contours[0])
                sum_this_worm_contour_areas += contour_area
                #print(contour_area)

        ave_worm_areas.append(sum_this_worm_contour_areas / framecount)
        #print(ave_worm_areas)

    #now calculate activity normalising over average area of that worm
    totalactivity = []

    for worms in range(0, wormcount):
        thiswormactivity = []

        if len(none_frames[worms]) == 0:
            chunk = [0,framecount-1]
            chunks = [chunk]
        else:
            chunks = find_missing_ranges(none_frames[worms], framecount-1)


        for chunk in chunks:
            # print(f"chunk: {chunk}")
            chunk_size = chunk[1] - chunk[0] + 1
            # print(f"chunk_size: {chunk_size}")
            modulo = (chunk_size - 1) % 4
            # print(f"modulo: {modulo}")
            this_worm_total_frames = activity_pics[worms]

            for images in range(chunk[0],chunk_size+chunk[0],4):
                # print(f"images: {images}")

                #the range function will work for all groups of 5 frames but if there are some left over this accounts for that:
                if images + 1 == chunk_size+chunk[0] - modulo:
                    # print(f"condition: {images + 1 == chunk_size - modulo}")
                    #if there are unused frames we calculate a truncated activity
                    if modulo > 0:
                        thispic0 = this_worm_total_frames[images]
                        for i in range(modulo):
                            thispici = this_worm_total_frames[images + i+1]
                            thispic0 = cv.add(thispic0, thispici)

                        imgray = cv.cvtColor(thispic0, cv.COLOR_BGR2GRAY)
                        # Looks at image contrast, gives threshold value for contour
                        ret, thresh = cv.threshold(imgray, 127, 255, 0)
                        # Lists contours
                        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                        contour_area = cv.contourArea(contours[0])
                        #print(contour_area)
                        activityindex = contour_area / ave_worm_areas[worms] - 1
                        #print(modulo * (ave_worm_areas[worms]))

                        for i in range(0, modulo + 1):
                            # print(f"append final activity")
                            thiswormactivity.append(round(activityindex/(modulo+1), 2))
                        #print(thiswormactivity)


                    else:
                        thiswormactivity.append(thiswormactivity[-1])

                    # finished with calculating activity - go to save section
                    break


                #print(images)
                thispic0 = this_worm_total_frames[images]
                thispic1 = this_worm_total_frames[images + 1]
                thispic2 = this_worm_total_frames[images + 2]
                thispic3 = this_worm_total_frames[images + 3]
                thispic4 = this_worm_total_frames[images + 4]


                newpic0 = cv.add(thispic0,thispic1,thispic2)
                newpic1 = cv.add(thispic3,thispic4)
                newpic = cv.add(newpic0,newpic1)


                imgray = cv.cvtColor(newpic, cv.COLOR_BGR2GRAY)
                #Looks at image contrast, gives threshold value for contour
                ret, thresh = cv.threshold(imgray,127,255,0)
                #Lists contours
                contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
                contour_area = cv.contourArea(contours[0])

                if contour_area != 0:
                    #previously divided by first frame area and rounded
                    #activityindex = round(cv.contourArea(contours[0])/cv.contourArea(contours1[0]) - 1,2)
                    #print(contour_area)
                    #print(4 * ave_worm_areas[worms])
                    activityindex = round((contour_area / ave_worm_areas[worms] - 1)/4, 2)

                if contour_area == 0:
                    activityindex = 0

                if activityindex > 2:
                    activityindex = 2

                # print(f"append 4 times {activityindex}")
                thiswormactivity.extend([activityindex, activityindex, activityindex, activityindex])
                #print(f"frame: {images},  worm: {worms},  activity: {activityindex}")

        for i in none_frames[worms]:
            thiswormactivity.insert(i, None)

        totalactivity.append(thiswormactivity)

    return totalactivity
    print("Activity done!")

def centroid_movement(wormcount, framecount, centroid_list):
    total_centroidmovement = []

    for worm in range(wormcount):
        this_worm_movement = []
        # iterate through frames
        for i in range(framecount - 1):
            if centroid_list[worm][i] is None or centroid_list[worm][i+1] is None:
                this_worm_movement.append(None)
            else:
                centroid_distance = math.dist(centroid_list[worm][i], centroid_list[worm][i + 1])
                this_worm_movement.append(round(centroid_distance,2))

        this_worm_movement.append(None) #to make the list an approproiate length there will be an arbitrary 0 at the end as this is a between frame measure not on frame
        total_centroidmovement.append(this_worm_movement)

    return total_centroidmovement
    print("Centroid Movement done!")

def movement_efficiency(wormcount, framecount, totalcentroidmovement, totalactivity):

    movement_efficiency_all_worms = []

    end_frames = framecount%4
    #print(end_frames)

    for worm in range(wormcount):
        movement_efficiency_this_worm = []
        activity = totalactivity[worm]
        movement = totalcentroidmovement[worm]
        #print(activity)
        #print(movement)

        for frame in range(framecount):
            if activity[frame] is None or movement[frame] is None:
                movement_efficiency_this_worm.append(None)

            else:
                if movement[frame] == 0:
                    movement_efficiency_value = 0
                elif activity[frame] != 0:
                    movement_efficiency_value = round(movement[frame] / activity[frame], 2)
                else:
                    movement_efficiency_value = 35 # when activity is 0 we need a cap so it is not infinity ( worm is travelling but there is no activity therefore maximum efficiency )

                movement_efficiency_this_worm.append(movement_efficiency_value)

        # this originally was written to calculate in groups of 4 but since None values can occur in a number of places for an number of reasons, it calculates the above on a frame by frame basis
        # for frame in range(0, framecount, 4):
        # #print(f"frame {frame}")
        #     if frame < framecount - end_frames:
        #         movement_four = movement[frame] +  movement[frame + 1] + movement[frame + 2] + movement[frame + 3]
        #         if activity[frame] != 0:
        #             movement_efficiency_value = round(movement_four / (activity[frame])*4, 2)
        #         else:
        #             activity[frame] = "inf"
        #             #print(f"value = {movement_efficiency_value}")
        #             movement_efficiency_this_worm.extend([movement_efficiency_value, movement_efficiency_value, movement_efficiency_value, movement_efficiency_value])
        #
        #     else:
        #         end_mvmt = 0
        #         for i in range(end_frames):
        #             end_mvmt += movement[framecount-end_frames + i]
        #         if activity[frame] != 0:
        #             movement_efficiency_value = round((end_mvmt/(activity[frame]) * end_frames), 2)
        #         else:
        #             activity[frame] = "inf"
        #         for i in range(end_frames):
        #             movement_efficiency_this_worm.append(movement_efficiency_value)

        movement_efficiency_all_worms.append(movement_efficiency_this_worm)

    return movement_efficiency_all_worms
    print("Movement Efficiency done!")

#-------------------- save parameters ------------------------------------------------------------------------------------

def save(curling, asymmetry, bend, stretch, activity, centroidmovement, movementefficiency, folder, wormcount):
    foldername = str(folder)
    if os.path.isdir(str(folder) + "/results") == False:
        os.mkdir(str(folder) + "/results")

    for worms in range(0, wormcount):

    #     #
    #     newactivity = activity[worms]
    #     newactivity = np.append(newactivity, (0))  # add a 0 to make it the right length, as with centroid movement this is a between frame measure

        dataset = pd.DataFrame(
            {'Curling': curling[worms], 'Asymmetry': asymmetry[worms],
             'Bend': bend[worms], 'Stretch': stretch[worms], 'Activity': activity[worms],
             'Centroid Movement': centroidmovement[worms], 'Movement Efficiency': movementefficiency[worms]},
            columns=['Curling', 'Asymmetry', 'Bend', 'Stretch', 'Activity', 'Centroid Movement',
                     'Movement Efficiency'])

        savename = str(folder) + "/results/parameters_" + str(worms) + ".csv"
        dataset.to_csv(savename)

def filter_none(data):
    return [x for x in data if x is not None]

def summary_statistics(curling, asymmetry, bend, stretch, activity, centroidmovement, movementefficiency, folder, wormcount):

    # Save a file with all worms, averages for each measure
    foldername = str(folder)
    if not os.path.isdir(foldername + "/results"):
        os.mkdir(foldername + "/results")

    percentagecurling = []
    asymmetryvector = []
    bendvector = []
    maxbendvector = []
    stretchvector = []
    maxstretchvector = []
    activityvector = []
    centroidmovementvector = []
    movementefficiencyvector = []

    for worms in range(wormcount):

        # Process each metric, filtering out None values
        filtered_curling = filter_none(curling[worms])
        filtered_asymmetry = filter_none(asymmetry[worms])
        filtered_bend = filter_none(bend[worms])
        filtered_stretch = filter_none(stretch[worms])
        filtered_activity = filter_none(activity[worms])
        filtered_centroidmovement = filter_none(centroidmovement[worms])
        filtered_movementefficiency = filter_none(movementefficiency[worms])
        # print(filtered_asymmetry, filtered_bend, filtered_stretch, filtered_activity, filtered_centroidmovement, filtered_movementefficiency)

        percentagecurling.append(round(filtered_curling.count(1)/len(filtered_curling), 3))
        asymmetryvector.append(round(sum(filtered_asymmetry) / len(filtered_asymmetry), 2))
        bendvector.append(round(sum(filtered_bend) / len(filtered_bend), 2))
        maxbendvector.append(round(np.max(filtered_bend), 2))
        stretchvector.append(round(sum(filtered_stretch) / len(filtered_stretch), 2))
        maxstretchvector.append(round(np.max(filtered_stretch), 2))
        activityvector.append(round(sum(filtered_activity) / len(filtered_activity), 2))
        centroidmovementvector.append(round(sum(filtered_centroidmovement) / len(filtered_centroidmovement), 2))
        movementefficiencyvector.append(round(sum(filtered_movementefficiency) / len(filtered_movementefficiency), 2))


    totalset = pd.DataFrame({
        'Percentage Curling': percentagecurling,
        'Average Asymmetry': asymmetryvector,
        'Average Bend': bendvector,
        'Maximum Bend': maxbendvector,
        'Average Stretch': stretchvector,
        'Maximum Stretch': maxstretchvector,
        'Average Activity': activityvector,
        'Average Centroid Movement': centroidmovementvector,
        'Average Movement Efficiency': movementefficiencyvector
    })

    savename = foldername + f"/results/summary_statistics.csv"
    totalset.to_csv(savename, index=False)
