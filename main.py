import os, time
import modules.hand_process as hp
import cv2
import autopy
import numpy as np

##########################
wCam, hCam = 640, 480
frameR = 75 # Frame Reduction
smoothening = 10
#########################
 
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
 
cap = cv2.VideoCapture(0)
detector = hp.Hand_Process(True)
wScr, hScr = autopy.screen.size()
print(wScr, hScr, wCam, hCam)
 
def box_check(x, y):
    return x > frameR and x < wCam - frameR and y > frameR and y < hCam - frameR

click_timer = 0 #click timer
clicked = False #click flag
while cap.isOpened():
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.process(img)
    
    if len(detector.fingers_pos) != 0:
        x0, y0 = detector.fingers_pos[4]
        x1, y1 = detector.fingers_pos[8] #index finger
        x2, y2 = detector.fingers_pos[12]#middle finger
    
    # 3. Check which fingers are up
    ups, dis = detector.open_fingers
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

    #mouse activate
    if len(ups) and box_check(x1, y1):
        #Moving Mode
        if ups[1]:
            # 5. Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            # 6. Smoothen Values
            clocX = plocX + (x3 - plocX) // smoothening
            clocY = plocY + (y3 - plocY) // smoothening

            # 7. Move Mouse
            autopy.mouse.move(clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY
        
        #Toggle Mode
        if len(ups) > 0 and ups[2] and box_check(x1, y1):
            cv2.circle(img, (x2, y2), 15, (0, 255, 255), cv2.FILLED)
            if pTime - click_timer > 0.5:
                autopy.mouse.toggle(autopy.mouse.Button.LEFT, True)
            click_timer = time.time() if not clicked else click_timer
            clicked = True

        if len(ups) > 0 and not ups[2] and box_check(x1, y1):
            if click_timer > 0:
                if pTime - click_timer > 0.5:
                    autopy.mouse.toggle(autopy.mouse.Button.LEFT, False)
                else:
                    autopy.mouse.click()
                    time.sleep(0.05)
            clicked = False


    # 11. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
    (255, 0, 0), 3)
    # 12. Display
    cv2.imshow("Image", img)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()