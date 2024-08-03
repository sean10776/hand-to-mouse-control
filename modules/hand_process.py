import cv2      
import mediapipe as mp
import numpy as np  

class Hand_Process:
    hand_tips = [1, 5, 9, 13, 17]
    tips = [4, 8, 12, 16, 20]
    def __init__(self, draw = False) -> None:
        self.hands = mp.solutions.hands.Hands(
                                    model_complexity=0,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)

        self.mpHand = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.draw = draw
        self.lmlist = []
        self.real_lm= []
        self._open_fingers = []
        self._fingers_dis = []

    def _finger_up(self):
        org = self.real_pos[0]
        self._fingers_dis = []
        self._open_fingers = []
        # _len, _deg = [], []
        for i in range(0, 5):
            theta = 0
            a = org - self.real_pos[self.hand_tips[i]]
            for j in range(self.hand_tips[i]+1, self.tips[i]+1):
                b = self.real_pos[j-1] - self.real_pos[j]
                a_bar, b_bar = np.linalg.norm(a), np.linalg.norm(b)     #3d len
                _theta = np.rad2deg(np.arccos(np.dot(a, b)/a_bar/b_bar))
                theta += _theta #3d ang[deg]
                a = b

            cond = theta < 100
            self._open_fingers.append(cond)
            if i > 0:
                if self.open_fingers[0][i-1] and self.open_fingers[0][i]:
                    ab = self.fingers_pos[self.tips[i-1]] - self.fingers_pos[self.tips[i]]
                    dis = np.linalg.norm(ab)
                    self._fingers_dis.append(dis)
                else:
                    self._fingers_dis.append(-1)
        #     _len.append(cond)
        #     _deg.append(theta)
        # print(_len, '\t', _deg)

    def process(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        self.lmlist = []
        self.real_lm= []
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            h, w, c = img.shape
            for id, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmlist.append([cx, cy])
                self.real_lm.append([lm.x , lm.y, lm.z])
            
            xmin, ymin = np.min(self.fingers_pos, axis=0)
            xmax, ymax = np.max(self.fingers_pos, axis=0)
            box_size = (ymax - ymin)*(xmax - xmin)
            img_size = h*w
    
            if self.draw:
                #draw joint
                self.mpDraw.draw_landmarks(img, hand,
                                            self.mpHand.HAND_CONNECTIONS)
                #draw hand window
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                                (0, 255, 0), 2)
            
            if box_size / img_size >0.02:
                self._finger_up()

        return img

    @property
    def real_pos(self):
        return np.array(self.real_lm)

    @property
    def fingers_pos(self):
        """
        finger pos at img
        """
        return np.array(self.lmlist)

    @property
    def open_fingers(self):
        """
        finger ups & finger distance
        """
        if len(self.lmlist) == 0: return [], []
        return self._open_fingers, self._fingers_dis

if __name__ == '__main__':
    hp = Hand_Process(True)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        img = cv2.flip(img,1)

        lm = hp.process(img)
        print(hp.open_fingers)

        cv2.imshow('MediaPipe Hands', img)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()