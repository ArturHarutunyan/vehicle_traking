#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""@author: kyleguan
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
from moviepy.editor import VideoFileClip
from collections import deque
from sklearn.utils.linear_assignment_ import linear_assignment
import time
import helpers
import detector
#import tracker
from tracker import Tracker
# Global variables to be used by funcitons of VideoFileClop
frame_count = 0 # frame counter
tracked_count = 0
max_age = 5  # no.of consecutive unmatched detection before 
             # a track is deleted
tracker_list = []
min_hits =2  # no. of consecutive matches needed to establish a track

tracker_list =[] # list for trackers
# list for track ID
track_id_list= deque(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'])

debug = False

def assign_detections_to_trackers(trackers, detections, iou_thrd = 0.3):
    '''
    From current list of trackers and new detections, output matched detections,
    unmatchted trackers, unmatched detections.
    '''    
    
    IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
    for t,trk in enumerate(trackers):
        #trk = convert_to_cv2bbox(trk) 
        for d,det in enumerate(detections):
         #   det = convert_to_cv2bbox(det)
            IOU_mat[t,d] = helpers.box_iou2(trk,det) 
    
    # Produces matches       
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)
    
    matched_idx = linear_assignment(-IOU_mat)        

    unmatched_trackers, unmatched_detections = [], []
    for t,trk in enumerate(trackers):
        if(t not in matched_idx[:,0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if(d not in matched_idx[:,1]):
            unmatched_detections.append(d)

    matches = []
   
    # For creating trackers we consider any detection with an 
    # overlap less than iou_thrd to signifiy the existence of 
    # an untracked object
    
    for m in matched_idx:
        if(IOU_mat[m[0],m[1]]<iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)       
    


def pipeline(img):
    '''
    Pipeline function for detection and tracking
    '''
    global frame_count
    global tracked_count

    global tracker_list
    global max_age
    global min_hits
    global track_id_list
    global debug
    global z_box
    
    img_dim = (img.shape[1], img.shape[0])
    
    if frame_count%10<2:
        z_box = det.get_localization(img) # measurement
        if len(z_box)>0:
            for i in range(len(z_box)):
                box = z_box[i]
                tmp_trk = Tracker() # Create a new tracker
                x = np.array([[box[0], 0, box[1], 0, box[2], 0, box[3], 0]]).T
                tmp_trk.x_state = x
                tmp_trk.predict_only()
                if len(tracker_list)!=0 and i<len(tracker_list):
                    tracker_list[i] = tmp_trk

                else:
                    tracker_list.append(tmp_trk)
                    print(tracker_list)
                img= helpers.draw_box_label(img, box) 
    else:
        if len(z_box)>0:
            print(tracker_list)

            nb = []
            for i in range(len(z_box)):
                z = z_box[i]
                z = np.expand_dims(z, axis=0).T
                tmp_trk= tracker_list[i]
                tmp_trk.kalman_filter(z)
                xx = tmp_trk.x_state.T[0].tolist()
                xx =[xx[0], xx[2], xx[4], xx[6]]
                nb.append(xx)
                img= helpers.draw_box_label(img, xx) 
            for i in range(len(nb)):
                print(nb[i])
                print(z_box[i])
            z_box = nb


    frame_count+=1

    if debug:
       print('Frame:', frame_count)
    
       
    return img
    
if __name__ == "__main__":    
    
    det = detector.CarDetector()
    """
    if debug: # test on a sequence of images
        images = [plt.imread(file) for file in glob.glob('./test_images/*.jpg')]
        
        for i in range(len(images))[0:7]:
             image = images[i]
             image_box = pipeline(image)   
             plt.imshow(image_box)
             plt.show()
           
    else: # test on a video file.
    """
    start=time.time()
    output = 'testa_v13.mp4'
    clip1 = VideoFileClip("project_video.mp4")#.subclip(4,49) # The first 8 seconds doesn't have any cars...
    clip = clip1.fl_image(pipeline)
    clip.write_videofile(output, audio=False)
    end  = time.time()
        
    print(round(end-start, 2), 'Seconds to finish')
