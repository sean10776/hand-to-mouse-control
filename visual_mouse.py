import time
import cv2      
import mediapipe as mp
import threading as th
import numpy as np

class visual_mosue(th.Thread):
    def __init__(self) -> None:
        th.Thread.__init__(self)
        # For webcam input:
        self.cap = None
        self.hands = mp.solutions.hands.Hands(
                                    model_complexity=0,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)
        self.start_flag = True

    def run(self) -> None:
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.detect()

    def stop(self):
        print('stop')
        self.start_flag = False
        self.join()

    def detect(self):
        timer = time.time()
        while self.cap.isOpened() and self.start_flag:
            success, image = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            image = cv2.flip(image,1)
            result = self.process(image)

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0].landmark
                index_fig = hand[5:9]
                length = self.to_len(index_fig[0], index_fig[-1])
                hand_size = self.hand_size(hand)
                print(length, '\t', hand_size)

            if time.time() - timer < 10:
                font = cv2.FONT_HERSHEY_SIMPLEX
                h = image.shape[0]
                cv2.putText(image, 'Please open your hand', (0,h-10), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
            time.sleep(0.01)
        self.cap.release()
        cv2.destroyAllWindows()

    def process(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.hands.process(img)

    def hand_size(self, lms):
        lm0 = lms[0]
        total = 0
        for i in [lms[5], lms[9], lms[13], lms[17]]:
            total += self.to_len(lm0, i)
        return total / 4

    def to_len(self, lm, lm2):
        pos = np.array([lm.x, lm.y, lm.z])
        pos2 = np.array([lm2.x, lm2.y, lm2.z])
        return np.sum((pos-pos2)**2)**0.5

if __name__ == "__main__":
    vm = visual_mosue()
    vm.start()
