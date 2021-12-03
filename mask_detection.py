import numpy as np
from imutils.video import VideoStream
import datetime
import cv2
import numpy as np
import argparse

from utilies import detector_utils

detection_graph, sess = detector_utils.load_inference_graph()
lst1=[]
lst2=[]
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--display', dest='display', type=int,
                        default=1, help='Display the detected images using OpenCV. This reduces FPS')
args = vars(ap.parse_args())

if __name__ == '__main__':
    score_thresh = 0.80
    vs = VideoStream(0).start()
    start_time = datetime.datetime.now()
    num_frames = 0
    im_height, im_width = (None, None)
    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
    def count_no_of_times(lst):
        x=y=cnt=0
        for i in lst:
            x=y
            y=i
            if x==0 and y==1:
                cnt=cnt+1
        return cnt
    try:
        while True:
            frame = vs.read()
            frame = np.array(frame)
            if im_height is None:
                im_height, im_width = frame.shape[:2]

            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                print("Error converting to RGB")
            boxes, scores, classes = detector_utils.detect_objects(
                frame, detection_graph, sess)
            a, b = detector_utils.draw_box_on_image(
                len(classes), score_thresh, scores, boxes, classes, im_width, im_height, frame)
            lst1.append(a)
            lst2.append(b)
            num_frames += 1
            elapsed_time = (datetime.datetime.now() -
                            start_time).total_seconds()
            fps = num_frames / elapsed_time
            if args['display']:
                # Display FPS on frame
                detector_utils.draw_text_on_image("FPS : " + str("{0:.2f}".format(fps)), frame)
                cv2.imshow('Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    vs.stop()
                    break
            print("Average FPS: ", str("{0:.2f}".format(fps)))
    except KeyboardInterrupt:
        print("Average FPS: ", str("{0:.2f}".format(fps)))


