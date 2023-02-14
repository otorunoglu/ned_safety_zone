import cv2
import numpy as np
import time
from multiprocessing import Process, Value
from pyniryo import *

def transparent(img):
    overlay = img.copy()
    height, width, channels = img.shape
    alpha = 0.2  # Transparency factor.
    # Following line overlays transparent rectangle over the image
    cv2.circle(overlay, center=(width//2, height),radius=width, color=(0,0,0), thickness=230)
    cv2.circle(overlay, center=(width//2, height),radius=width//100*53, color=(255,0,0),thickness=-1,lineType= cv2.LINE_AA)
    image_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.circle(overlay, center=(width//2, height),radius=width//100*64, color=(0,0,255), thickness=125)
    cv2.circle(overlay, center=(width//2, height),radius=width//100*84, color=(0,255,0), thickness=110)

    image_new = cv2.addWeighted(overlay, alpha, image_new, 1 - alpha, 0)
    return image_new 

def stop_zone(height, width,img_rgb):
    cv2.circle(img_rgb, center=(width//2, height),radius=width, color=(0,0,0), thickness=450)
    cv2.circle(img_rgb, center=(width//2, height),radius=width//100*53, color=(0,0,0),thickness=-1,lineType= cv2.LINE_AA)
    return img_rgb

def alarm_zone(height, width, img_rgb):
    cv2.circle(img_rgb, center=(width//2, height),radius=width//100*74, color=(0,0,0),thickness=-1,lineType= cv2.LINE_AA)
    cv2.circle(img_rgb, center=(width//2, height),radius=width, color=(0,0,0), thickness=230)
    points1 = np.array([[1,622],[29,290], [99,1], [2,0]])
    points2 = np.array([[668,622],[635,280], [561,1], [669,0]])
    cv2.fillPoly(img_rgb, pts=[points1], color=(0, 0, 0))
    cv2.fillPoly(img_rgb, pts=[points2], color=(0, 0, 0))
    return img_rgb

def motion_detector_gist(is_motion_detected):
        camera = cv2.VideoCapture(0)
        previous_frame = None
        i=0        
        while True:
            _,img = camera.read()
            img = img[50:688, 560:1230]
            height, width, channels = img.shape       

            if i==0:
                _,original = camera.read()  
                original = original[50:688, 560:1230]
                original_rgb = cv2.cvtColor(src=original, code=cv2.COLOR_BGR2RGB)
                cv2.circle(original_rgb, center=(width//2, height),radius=width//100*53, color=(0,0,0),thickness=-1,lineType= cv2.LINE_AA)

                prepared_original = cv2.cvtColor(original_rgb, cv2.COLOR_BGR2GRAY)
                prepared_original = cv2.GaussianBlur(src= prepared_original, ksize=(5, 5), sigmaX=0)
                i+=1

            
            _,alarm   = camera.read()
            alarm = alarm[50:688, 560:1230]
            _,stop   = camera.read()
            stop = stop[50:688, 560:1230]
            image_new= transparent(img)           
            alarm =   alarm_zone(height,width,alarm)
            stop =  stop_zone(height,width,stop)

            # 1. Load image; convert to RGB
            img_rgb = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
            cv2.circle(img_rgb, center=(width//2, height),radius=width//100*53, color=(0,0,0),thickness=-1,lineType= cv2.LINE_AA)

            #img_rgb1 = stop_zone(height, width,img_rgb)
            #img_rgb2 = alarm_zone(height, width,img_rgb)
            

            # 2. Prepare image; grayscale and blur
            
            prepared_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5, 5), sigmaX=0)

            # 2. Calculate the difference
            if (previous_frame is None  ): #and previous_frame2 is None
            # First frame; there is no previous one yet
                previous_frame = prepared_frame
                continue

            # calculate difference and update previous frame
            diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
            previous_frame = prepared_frame

            diff_frame_orj = cv2.absdiff(src1= prepared_original, src2=prepared_frame)

            # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
            kernel = np.ones((5, 5))
            diff_frame = cv2.dilate(diff_frame, kernel, 1)
            diff_frame_orj = cv2.dilate(diff_frame_orj, kernel, 1)

            # 5. Only take different areas that are different enough (>20 / 255)
            thresh_frame_org_1 = cv2.threshold(src=diff_frame_orj, thresh=50, maxval=255, type=cv2.THRESH_BINARY)[1]
            thresh_frame_org_2 = cv2.threshold(src=diff_frame_orj, thresh=50, maxval=255, type=cv2.THRESH_BINARY)[1]
            thresh_frame_org_1 = stop_zone(height,width,thresh_frame_org_1)
            thresh_frame_org_2 = alarm_zone(height,width,thresh_frame_org_2)

            thresh_frame_1 = cv2.threshold(src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]
            thresh_frame_2=cv2.threshold(src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]
            thresh_frame_1 = stop_zone(height,width,thresh_frame_1)
            thresh_frame_2 = alarm_zone(height,width,thresh_frame_2)

            # 6. Find and optionally draw contours
            diff_frame = cv2.dilate(diff_frame, kernel, 1)
            contours1, _ = cv2.findContours(image=thresh_frame_1, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            contours2, _ = cv2.findContours(image=thresh_frame_2, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            contours_org1, _ = cv2.findContours(image=thresh_frame_org_1, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            contours_org2, _ = cv2.findContours(image=thresh_frame_org_2, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

            # Comment below to stop drawing contours
            cv2.drawContours(image=image_new, contours=contours1, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.drawContours(image=image_new, contours=contours2, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)



            # 7.detection control

            if len(contours1) > 0 or len(contours_org1) > 0:
                print("Motion detected in the stop zone")
                is_motion_detected.value = 0
            elif len(contours2) > 0 or len(contours_org2) > 0:
                print("Motion detected in the alarm zone")
                is_motion_detected.value = 0.05
            else:
                is_motion_detected.value = 0.1
                print("NOT")

            cv2.imshow('Motion detector', image_new)        
            if (cv2.waitKey(30) == 27):
                break
            
        # Cleanup
        camera.release()
        cv2.destroyAllWindows()

    
   
def robot_motion():
        
        # create a Value object to store the motion detected status
        motion_detected = Value('f', 0.1)
    
        # create a process to run the motion detector function
        process = Process(target=motion_detector_gist, args=(motion_detected,))
        process.start()
    
        robot = NiryoRobot("169.254.200.200") # 169.254.200.200 - 127.0.0.1
        
        robot.calibrate_auto()
        robot.update_tool()
        robot.move_joints(0.0, -0.60, 0.0, 0.0, 0.0, 0.0)
        j1, j2, j3, j4, j5, j6 =  robot.get_joints() 

        while True:
            while j1<1.8:
                # if motion_detected.value==1:
                #     jog1=0
                # elif motion_detected.value==2:
                #     jog1=0.05
                # else:
                #     jog1=0.1
                time.sleep(0.1)
                robot.jog_joints([motion_detected.value, 0, 0, 0, 0, 0])
                j1, j2, j3, j4, j5, j6 =  robot.get_joints()
                time.sleep(0.1)

            while j1>-1.8:          
                # if motion_detected.value==1:
                #     jog1=-0.000001
                # elif motion_detected.value==2:
                #     jog1=-0.01
                # else:
                #     jog1=-0.1
                # time.sleep(0.1)

                robot.jog_joints([-1*motion_detected.value, 0, 0, 0, 0, 0])
                j1, j2, j3, j4, j5, j6 =  robot.get_joints()
                time.sleep(0.1)
  

if __name__ == '__main__':
    robot_motion()