import cv2
from threading import Thread
from goprocam import GoProCamera
from goprocam import constants

class Webcam:

    def __init__(self):
        # self.gpCam = GoProCamera.GoPro()
        # self.gpCam.gpControlSet(constants.Stream.BIT_RATE, constants.Stream.BitRate.B2_4Mbps)
        # self.video_capture = cv2.VideoCapture("udp://127.0.0.1:10000")
        self.video_capture = cv2.VideoCapture(0)
        self.current_frame = self.video_capture.read()[1]

    # create thread for capturing images
    def start(self):
        self.th = Thread(target=self._update_frame, args=())
        self.th.daemon = True
        self.th.start()

    def _update_frame(self):
        while(True):
            self.current_frame = self.video_capture.read()[1]

    # get the current frame
    def get_current_frame(self):
        return self.current_frame

    def end(self):
        self.video_capture.release()
        cv2.destroyAllWindows()
