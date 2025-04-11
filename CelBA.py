import tkinter as tk
from tkinter import filedialog, PhotoImage
from tkinter.ttk import *
from PIL import ImageTk, Image
import cv2
import os
import CelBA_SubFunctions
from threading import Thread
import numpy as np



###Root
root = tk.Tk()
root.title("CelBA")

icon = PhotoImage(file = '/Users/caitlincutts/Documents/Cam/4th Yr/Project/CelBAfinal_zipup/CelBA_icon.png')
root.iconphoto(False, icon)

root.geometry("1300x1080")

tabControl = Notebook(root) 
  
tab1 = Frame(tabControl) 
tab2 = Frame(tabControl) 
tab3 = Frame(tabControl) 
tab4 = Frame(tabControl)
  
tabControl.add(tab1, text ='Select files') 
tabControl.add(tab2, text ='Contours')
tabControl.add(tab3, text ='Data Visualisation')
tabControl.pack(expand = 1, fill ="both")





# -----------------------------------------------tab1 = set up and data processing------------------------------------------

# Not yet put into GUI but editable from python file: (minimum val is 5 due to activity measure)
threshold = 5

Openfolder = Button(tab1,text = "Select folder",command=lambda:open_folder())

size_text = Label(tab1, text = "Size:")
resolution_text = Label(tab1, text = "Resolution:")
size_entry = tk.Entry(tab1, width=10, textvariable=tk.StringVar(value="200"))
resolution_entry = tk.Entry(tab1, width=10, textvariable=tk.StringVar(value="700"))

foldernamevar = "No folder selected"
Foldername = Label(tab1, text = foldernamevar)

overlap_progress_text = ""
Overlap_progress_text = tk.Label(tab1, text=overlap_progress_text)
Overlap_progress_text.place(x=10, y=210)

def open_folder():
    global foldernamevar, overlap_calculated
    file = filedialog.askdirectory()
    foldernamevar = file
    Foldername.configure(text=os.path.basename(foldernamevar))
    overlap_calculated = False

def process_button_clicked():
    # Download the file in a new thread.
    Thread(target=process).start()

def process():
    global framecount, wormcount, centroid_list, GUI_images, skeleton_table, numendstable, activity_pics

    if overlap_calculated:
        bool_list = overlap.bool_list
        start = [0]
        end = []
        for i in range(1,len(bool_list)):
            if bool_list[i] == False and bool_list[i - 1] == True: # TRUE > FALSE
                start.append(i)
                end.append(i-1)
            if bool_list[i] == True and bool_list[i - 1] == False: # FALSE > TRUE
                end.append(i-1)
                start.append(i)
        end.append(len(bool_list)) # If the final chunk goes to the end of the frames include the number of frames.
        swappable_chunks = []
        for i in range(len(start)):
            swappable_chunks.append(f"{start[i]}, {end[i]}") # End always calculates the end of the very first chunk so ignore this and only look at chunks after an overlap.
    else:
        bool_list = [False]
        swappable_chunks = "no_switches_detected"
    framecount, wormcount, centroid_list, GUI_images, skeleton_table, numendstable, activity_pics = CelBA_SubFunctions.worm_tracking(foldernamevar, int(size_entry.get()), bool_list, contourstatus)
    print("Images Processed!")
    update_wormselect()
    w.configure(to=len(GUI_images)-1)
    SwitchwormA.config(values=list(np.arange(1, wormcount + 1)))
    SwitchwormB.config(values=list(np.arange(1, wormcount + 1)))
    Framestoswitch.config(values=swappable_chunks)
    overlap_progress_text = "Images Processed!"
    Overlap_progress_text.configure(text=overlap_progress_text)

def size_button_clicked():
    # Download the file in a new thread.
    Thread(target=size_button).start()

def size_button():
    print(f"size being tested = {size_entry.get()}")
    global initial_images
    global initial_wormcount
    initial_images, initial_wormcount, first_frame_areas, first_frame_centroids = CelBA_SubFunctions.find_first_frame_contours(foldernamevar, size_entry.get())
    size_bar.configure(to_=initial_wormcount)
    initial_wormcount_value = str(initial_wormcount)
    wormcount_value.configure(text=initial_wormcount_value)
    root.update_idletasks()
    setimage_vis_size()

def overlap_button_clicked():
    # Download the file in a new thread.
    Thread(target=overlap_button).start()

def overlap_button():
    overlap_progress_text = "Beginning overlap detection."
    Overlap_progress_text.configure(text=overlap_progress_text)
    overlap_canvas.delete('all')
    overlap_canvas_noedit.delete('all')
    global overlap_images, overlap_frames, overlap_chunks, framecount, overlap
    images, overlap_wormcount, this_frame_areas, centroid_list = CelBA_SubFunctions.find_first_frame_contours(foldernamevar, size_entry.get())
    run_overlaps = CelBA_SubFunctions.look_for_overlaps(foldernamevar, int(size_entry.get()), threshold, overlap_wormcount)
    overlap_images = run_overlaps.traced_images
    overlap_frames = run_overlaps.overlap_frames
    overlap_chunks = run_overlaps.overlap_chunk
    framecount = len(overlap_images)
    overlap_bar.configure(to_=framecount)
    setimage_vis_overlap()
    if not any(overlap_frames):
        overlap_progress_text = "No overlaps detected, images are ready to process."
    else:
        overlap_progress_text = "Overlaps detected."
    Overlap_progress_text.configure(text=overlap_progress_text)
    root.update_idletasks()
    # CelBA_SubFunctions.draw_true_sections(overlap_chunks, overlap_canvas, framecount)
    CelBA_SubFunctions.draw_true_sections(overlap_frames, overlap_canvas_noedit, framecount)
    overlap = CelBA_SubFunctions.EditableRectangles(overlap_canvas, overlap_chunks, framecount)
    global overlap_calculated
    overlap_calculated = True
    print(overlap_calculated)

def contourstatus(count,step, total_data):

    if count == 0:
        # Set the maximum value for the progress bar.
        contourbar.configure(maximum=total_data)
    else:
        # Increase the progress.
        contourbar.step(step)

# def skeletonstatus(count,step, total_data):
#
#     if count == 0:
#         # Set the maximum value for the progress bar.
#         skeletonbar.configure(maximum=total_data)
#     else:
#         # Increase the progress.
#         skeletonbar.step(step)
#
# def overlaystatus(count,step, total_data):
#
#     if count == 0:
#         # Set the maximum value for the progress bar.
#         overlaybar.configure(maximum=total_data)
#     else:
#         # Increase the progress.
#         overlaybar.step(step)r


Process = Button(tab1,text = "Click to process",command=lambda:process_button_clicked())
Size_button = Button(tab1, text="Click to test size", command=lambda:size_button_clicked())
Overlap_button = Button(tab1, text="Click to detect overlaps", command=lambda:overlap_button_clicked())

contourbar = Progressbar(tab1)
contourtext = Label(tab1,text="Building contours:")
# skeletonbar = Progressbar(tab1)
# skeletontext = Label(tab1,text="Building skeletons:")
# overlaybar = Progressbar(tab1)
# overlaytext = Label(tab1,text="Building overlays:")

Openfolder.place(x=10, y=5)
Foldername.place(x=150, y=10)

size_text.place(x=10, y=50)
size_entry.place(x=150, y=40)

resolution_text.place(x=10, y=80)
resolution_entry.place(x=150, y=70)

Process.place(x=10, y=110)
Size_button.place(x=150, y=110)
Overlap_button.place(x=295, y=110)

# Progress bars
contourtext.place(x=10, y=180)
contourbar.place(x=150, y=180)

# skeletontext.place(x=10, y=210)
# skeletonbar.place(x=150, y=210)
#
# overlaytext.place(x=10, y=240)
# overlaybar.place(x=150, y=240)

# ------------------------------------------------------------------------------------------------------------------------------
# making it so you can test the size, overlap and see results

frame = 0
size_image = Image.fromarray(cv2.imread("blank.jpg"))
rescaled_size_image = size_image.resize((500,500), Image.Resampling.LANCZOS)
size_vis = ImageTk.PhotoImage(rescaled_size_image)
size_vis_label = Label(tab1,image=size_vis)
size_vis_label.place(x=500,y=50)

initial_wormcount = 1

def setimage_vis_size():
    image = Image.fromarray(initial_images[0])
    img = image.resize((500,500), Image.Resampling.LANCZOS)
    image1 = ImageTk.PhotoImage(img)
    size_vis_label.configure(image=image1)
    size_vis_label.image = image1

def setimage_vis_overlap():
    val = int(round(float(overlap_bar.get()),2)) -1
    image = Image.fromarray(overlap_images[val])
    img = image.resize((500,500), Image.Resampling.LANCZOS)
    image1 = ImageTk.PhotoImage(img)
    size_vis_label.configure(image=image1)
    size_vis_label.image = image1

def updatesize_vis(val):
    val = int(round(float(val),2))
    # print("value", val)
    # print(len(initial_images))
    image = Image.fromarray(initial_images[val-1])
    img = image.resize((500,500), Image.Resampling.LANCZOS)
    image1 = ImageTk.PhotoImage(img)
    size_vis_label.configure(image = image1)
    size_vis_label.image = image1

def updateoverlap_vis(val):
    val = int(round(float(val),2))
    image = Image.fromarray(overlap_images[val-1])
    img = image.resize((500,500), Image.Resampling.LANCZOS)
    image1 = ImageTk.PhotoImage(img)
    size_vis_label.configure(image = image1)
    size_vis_label.image = image1

framecount = 1
overlap_bar = tk.Scale(tab1, from_=1, to=framecount, command=updateoverlap_vis, orient="horizontal", length=500)
overlap_bar.place(x=500, y=580)

size_bar = tk.Scale(tab1, from_=1, to=initial_wormcount, command=updatesize_vis, orient="horizontal", length=100)
size_bar.place(x=500,y =0)

initial_wormcount_value = ""
wormcount_text = tk.Label(tab1, text = "Wormcount:")
wormcount_value = tk.Label(tab1, text = initial_wormcount_value)
wormcount_text.place(x=650, y=10)
wormcount_value.place(x=750, y=10)


overlap_canvas = tk.Canvas(tab1, bg="white", width=500, height=30)
overlap_canvas_noedit = tk.Canvas(tab1, bg="white", width=500, height=20)

overlap_canvas.place(x=500,y =625)
overlap_canvas_noedit.place(x=500,y =665)








# -----------------------------------------------tab2 = params and vis processed------------------------------------------

#Moving through images 
#Tab 2
resolution = 700

# filenamevar = "No folder selected"
# Filename = Label(tab2, text = filenamevar)
# Filename.place(x=805,y=10)

image = Image.fromarray(cv2.imread("blank.jpg"))
img = image.resize((resolution,resolution), Image.Resampling.LANCZOS)
imageblank = ImageTk.PhotoImage(img)

myLabel = Label(tab2,image=imageblank)
myLabel.grid(row=2,column=0)

# def open_file():
#     imageseries = GUI_images
#     w.config(to_=len(imageseries -1))
#     w.update_idletasks()

# Openfile = Button(tab2,text = "Choose folder with Tab1 completed",command=lambda:[open_file(),update_wormselect()])
# Openfile.grid(row=0,column=0)

imageseries = 750

def setimage():
    global firstname
    global wormnumber
    wormnumber = int(Wormselect.get()) -1
    # firstname = relative_path + str(wormnumber) + "/0.png"
    val = int(round(float(w.get()), 2))
    firstname = GUI_images[val][wormnumber]
    image = Image.fromarray(firstname)
    img = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
    image1 = ImageTk.PhotoImage(img)
    myLabel.configure(image=image1)
    myLabel.image = image1

def updateImg(val):
    global GUI_images
    val = int(round(float(val),2))
    # imagename = firstname.replace("0.png","")
    # imagenew = imagename + str(val) + ".png"
    image = Image.fromarray(GUI_images[val][wormnumber])
    img = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
    image = ImageTk.PhotoImage(img)
    myLabel.configure(image = image)
    myLabel.image = image


    # Update the parameters in the GUI for this worm
    global curl_table, total_curvature, totalasymmetry, totalbend, totalstretch, totalactivity, totalcentroidmovement, totalme

    currentcurl = "Curling = " + str(curl_table[wormnumber][w.get()])
    currentkappa = "Curvature = " + str(total_curvature[wormnumber][w.get()])
    currentasymmetry = "Asymmetry = " + str(totalasymmetry[wormnumber][w.get()])
    currentbend = "Bend = " + str(totalbend[wormnumber][w.get()])
    currentstretch = "Stretch = " + str(totalstretch[wormnumber][w.get()])
    currentactivity = "Activity = " + str(totalactivity[wormnumber][w.get()])
    currentcentroidmovement = "Centroid Movement = " + str(totalcentroidmovement[wormnumber][w.get()])
    currentmovementefficiency = "Movement Efficiency = " + str(totalme[wormnumber][w.get()])

    Curlingtext.configure(text=currentcurl)
    Kappatext.configure(text=currentkappa)
    Asymmetrytext.configure(text=currentasymmetry)
    Bendtext.configure(text=currentbend)
    Stretchtext.configure(text=currentstretch)
    Activitytext.configure(text=currentactivity)
    Centroidmovementtext.configure(text=currentcentroidmovement)
    Movementefficiencytext.configure(text=currentmovementefficiency)


w = tk.Scale(tab2, from_=0, to_=imageseries, orient="horizontal", command=updateImg,length=500)
w.grid(row=1,column = 0)


Wormselect = Combobox(tab2,state="readonly",values=[])

def update_wormselect():
    Wormselect.config(values=list(np.arange(1, wormcount + 1)))

Wormselect.place(x=805,y=30)

Updateworm = Button(tab2,text="Update worm",command=setimage)
Updateworm.place(x=950,y=28)


def fix_switching():
    wormA = int(SwitchwormA.get()) -1
    wormB = int(SwitchwormB.get()) -1
    # how to get index
    placeholder = eval(Framestoswitch.get())

    start = placeholder[0]
    end = placeholder[1]

    global centroid_list, GUI_images, skeleton_table, numendstable
    for frame in range(start, end + 1):
        # Swap the images between wormA and wormB
        GUI_images[frame][wormA], GUI_images[frame][wormB] = GUI_images[frame][wormB], GUI_images[frame][wormA]
        skeleton_table[frame][wormA], skeleton_table[frame][wormB] = skeleton_table[frame][wormB], skeleton_table[frame][wormA]
        numendstable[wormA], numendstable[wormB] = numendstable[wormB], numendstable[wormA]
        centroid_list[wormA], centroid_list[wormB] = centroid_list[wormB], centroid_list[wormA]
    # numendstable = CelBA_SubFunctions,CelBA_SubFunctions.list_switch(numendstable, wormA, wormB, start, end)
    # centroid_list = CelBA_SubFunctions.list_switch(centroid_list, wormA, wormB, start, end)
    myLabel.update_idletasks()


SwitchwormA = Combobox(tab2,state="readonly",values=[], width=10)
SwitchwormB = Combobox(tab2,state="readonly",values=[], width=10)
Framestoswitch = Combobox(tab2,state="readonly",values=[], width=20)

SwitchwormA.place(x=900,y=520)
SwitchwormB.place(x=900,y=550)
Framestoswitch.place(x=900,y=580)

wormatext = Label(tab2, text = "Worm A:")
wormbtext = Label(tab2, text = "Worm B:")
frametext = Label(tab2, text = "Frames to switch tracking ID:")
wormatext.place(x=840,y=520)
wormbtext.place(x=840,y=550)
frametext.place(x=715,y=580)
Fixswitchingbutton = Button(tab2,text="Click to switch tracking",command=fix_switching)
Fixswitchingbutton.place(x=900,y=610)


#Curlingstuff

def paramsclick():
    Thread(target=params).start()

def params():
    #Curling

    global framecount, curl_table, total_curvature, totalasymmetry, totalbend, totalstretch, totalactivity, totalcentroidmovement, totalme, bodylengths, percentcurling

    #Kappa
    total_curvature, bodylengths = CelBA_SubFunctions.kappa(wormcount, framecount, skeleton_table)
    print("Kappa done")
    currentkappa = "Curvature = " + str(total_curvature[wormnumber][w.get()])
    Kappatext.configure(text=currentkappa)

    curl_table = CelBA_SubFunctions.count_curling(wormcount, framecount, numendstable, bodylengths)
    print("Curling done")
    # loadnamecurl = file + "/skeletons/frameset.npy"
    currentcurl = "Curling = " + str(curl_table[wormnumber][w.get()])
    Curlingtext.configure(text=currentcurl)

    totalasymmetry, totalbend, totalstretch = CelBA_SubFunctions.calc_curvature_params(wormcount, framecount, total_curvature)

    currentasymmetry = "Asymmetry = " + str(totalasymmetry[wormnumber][w.get()])
    Asymmetrytext.configure(text=currentasymmetry)
    currentbend = "Bend = " + str(totalbend[wormnumber][w.get()])
    Bendtext.configure(text=currentbend)
    currentstretch = "Stretch = " + str(totalstretch[wormnumber][w.get()])
    Stretchtext.configure(text=currentstretch)

    #Activity
    totalactivity = CelBA_SubFunctions.activity(wormcount, framecount, activity_pics)
    print("Activity done!")
    currentactivity = "Activity = " + str(totalactivity[wormnumber][w.get()])
    Activitytext.configure(text=currentactivity)

    #Centroid Movement
    totalcentroidmovement = CelBA_SubFunctions.centroid_movement(wormcount, framecount, centroid_list)
    print("Centroid Movement done!")
    currentcentroidmovement = "Centroid Movement = " + str(totalcentroidmovement[wormnumber][w.get()])
    Centroidmovementtext.configure(text=currentcentroidmovement)

    #Movement Efficiency
    totalme = CelBA_SubFunctions.movement_efficiency(wormcount, framecount, totalcentroidmovement, totalactivity)
    print("Movement efficiency done!")
    currentmovementefficiency = "Movement Efficiency = " + str(totalme[wormnumber][w.get()])
    Movementefficiencytext.configure(text=currentmovementefficiency)



    # Batch_parameters.save(file, 0, wormcount)
    # Batch_parameters.summary_statistics(file,0, wormcount)
    # print("Results saved")


    # #Movement
    # Batch_parameters.movement(file)
    # print("Movement done!")
    # loadnamemovement = file + "/skeletons/totalmovement.npy"
    # currentmovement = "Movement = " + str(np.load(loadnamemovement,allow_pickle=True)[wormnumber][w.get()])
    # Movementtext.configure(text=currentmovement)
    #Turning points
    # Batch_parameters.turningpointarray(file, wormcount)
    # print("Turning points done")
    # loadnametp = file + "/skeletons/totalturningpoints.npy"
    # currenttp = "Turning points = " + str(np.load(loadnametp,allow_pickle=True)[wormnumber][w.get()])
    # Tptext.configure(text=currenttp)
    #
    # #Wave number
    # Batch_parameters.bodywavenumber(wormcount, framecount, total_curvature)
    # print("Wave number done")
    # loadnamewavenumber = file + "/skeletons/totalwavenumber.npy"
    # currentwavenumber = "Wave number = " + str(np.load(loadnamewavenumber,allow_pickle=True)[wormnumber][w.get()])
    # Wavenumbertext.configure(text=currentwavenumber)
    #
    # #Movement distance
    # Batch_parameters.movementdistance(file)
    # print("Movement distance done!")
    # loadnamemovementdistance = file + "/skeletons/totalmovementdistance.npy"
    # currentmovementdistance = "Movement distance (mm) = " + str(np.load(loadnamemovementdistance,allow_pickle=True)[wormnumber][w.get()])
    # Movementdistancetext.configure(text=currentmovementdistance)
    #
    # #Movement speed
    # Batch_parameters.movementspeed(file)
    # print("Movement speed done!")
    # loadnamemovementspeed = file + "/skeletons/totalmovementspeed.npy"
    # currentmovementspeed = "Movement speed (mm/s) = " + str(np.load(loadnamemovementspeed,allow_pickle=True)[wormnumber][w.get()])
    # Movementspeedtext.configure(text=currentmovementspeed)


Calculateparams = Button(tab2,text="Calculate parameters",command=paramsclick)
Calculateparams.place(x=1033,y=28)

currentcurl = "No parameters calculated!"
Curlingtext = Label(tab2,text=currentcurl)
Curlingtext.place(x=1030,y=65)


currentkappa = "No parameters calculated!"
Kappatext = Label(tab2,text=currentkappa)
Kappatext.place(x=1030,y=85)

currenttp = "No parameters calculated!"
Tptext = Label(tab2,text=currenttp)
# Tptext.place(x=1030,y=105)

currentwavenumber = "No parameters calculated!"
Wavenumbertext = Label(tab2,text=currentwavenumber)
# Wavenumbertext.place(x=1030,y=125)

currentasymmetry = "No parameters calculated!"
Asymmetrytext = Label(tab2,text=currentasymmetry)
Asymmetrytext.place(x=1030,y=145)

currentbend = "No parameters calculated!"
Bendtext = Label(tab2,text=currentbend)
Bendtext.place(x=1030,y=165)

currentstretch = "No parameters calculated!"
Stretchtext = Label(tab2,text=currentstretch)
Stretchtext.place(x=1030,y=185)

currentactivity = "No parameters calculated!"
Activitytext = Label(tab2,text=currentactivity)
Activitytext.place(x=1030,y=205)

currentmovement = "No parameters calculated!"
Movementtext = Label(tab2,text=currentmovement)
# Movementtext.place(x=1030,y=225)

currentcentroidmovement = "No parameters calculated!"
Centroidmovementtext = Label(tab2,text=currentcentroidmovement)
Centroidmovementtext.place(x=1030,y=245)

currentmovementefficiency = "No parameters calculated!"
Movementefficiencytext = Label(tab2,text=currentmovementefficiency)
Movementefficiencytext.place(x=1030,y=265)

currentmovementdistance = "No parameters calculated!"
Movementdistancetext = Label(tab2,text=currentmovementdistance)
# Movementdistancetext.place(x=1030,y=285)

currentmovementspeed = "No parameters calculated!"
Movementspeedtext = Label(tab2,text=currentmovementspeed)
#Movementspeedtext.place(x=1030,y=305)

Save_button = Button(tab2, text="Save", command=lambda:save_button_clicked())
Save_button.place(x=1050,y=320)

def save_button_clicked():
    global curl_table,  totalasymmetry, totalbend, totalstretch, totalactivity, totalcentroidmovement, totalme, foldernamevar, wormcount
    CelBA_SubFunctions.save(curl_table,  totalasymmetry, totalbend, totalstretch, totalactivity, totalcentroidmovement, totalme, foldernamevar, wormcount)
    CelBA_SubFunctions.summary_statistics(curl_table, totalasymmetry, totalbend, totalstretch, totalactivity, totalcentroidmovement, totalme, foldernamevar, wormcount)
    save_text = "Save Complete!"
    Save_text.configure(text=save_text)

save_text = ""
Save_text = tk.Label(tab2, text=save_text)
Save_text.place(x=1050, y=350)


# -----------------------------------tab3 = Data vis screen--------------------------------------------------------------
#for opening up processed images and checking for worm overlaps, seeing if the processing has had desired result, checking footage etc

file_to_vis_var = "No folder selected"
Filename_vis = Label(tab3, text = file_to_vis_var)
Filename_vis.place(x=0,y=0)

frame = 0
image = Image.fromarray(cv2.imread("blank.jpg"))
img = image.resize((resolution,resolution), Image.Resampling.LANCZOS)
imageblank_vis = ImageTk.PhotoImage(img)
myLabel_vis = Label(tab3,image=imageblank_vis)
myLabel_vis.grid(row=2,column=0)

def open_file_vis():
    global imageseries
    global file
    global file_path
    global wormcount
    file = filedialog.askdirectory()
    file_path = sorted(os.listdir(file))
    Filename_vis.configure(text=file)
    imageseries = len(file_path) -1
    z.configure(to_=imageseries)

Openfile_vis = Button(tab3,text = "Select folder with image series to visualise",command=lambda:[open_file_vis(), setimage_vis()])
Openfile_vis.grid(row=0,column=10)

def setimage_vis():
    global firstname
    firstname = file_path[0]
    image = Image.fromarray(cv2.imread(os.path.join(file, firstname)))
    img = image.resize((resolution,resolution), Image.Resampling.LANCZOS)
    image1 = ImageTk.PhotoImage(img)
    myLabel_vis.configure(image=image1)
    myLabel_vis.image = image1

def updateImg_vis(val):
    val = int(round(float(val),2))
    image = Image.fromarray(cv2.imread(os.path.join(file, file_path[val])))
    img = image.resize((resolution,resolution), Image.Resampling.LANCZOS)
    image = ImageTk.PhotoImage(img)
    myLabel_vis.configure(image = image)
    myLabel_vis.image = image


z = tk.Scale(tab3, from_=0, to_=imageseries, orient="horizontal", command=updateImg_vis,length=500)
z.grid(row=1,column = 0)

###Event Loop
root.mainloop()

