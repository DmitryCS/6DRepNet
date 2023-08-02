import argparse

import cv2, numpy as np

from helpers import get_faces, get_angles, draw_line, draw_text, draw_box, draw_axis

parser = argparse.ArgumentParser(description='Process 6DRepNet.')
parser.add_argument('-i', '--image', default='', type=str, help='path to image')
parser.add_argument('-v', '--video', default='', type=str, help='path to video')
parser.add_argument('--infer', default='triton', type=str, help='head pose infer: torch | triton')

def process_frame(frame, frame_width, frame_height):
    boxes = get_faces(frame)
    for box in boxes:
        box = (box * np.array([frame_width, frame_height, frame_width, frame_height], dtype=np.float32)).astype(np.int32)
        img_face = frame[box[1]:box[3], box[0]:box[2]]
        if img_face.shape[0] > 5 and img_face.shape[1] > 5:
            roll, pitch, yaw = get_angles(img_face)
            print('roll, pitch, yaw:', roll, pitch, yaw)
            # draw_axis(frame, yaw, pitch, roll, box[0], box[1])
            draw_box(frame, box[0], box[1], box[2], box[3])
            draw_text(frame, yaw[0], pitch[0], roll[0], box[0], box[1])
            draw_line(frame, yaw, pitch, roll, (box[0]+box[2]) // 2, (box[1]+box[3]) // 2)
    return frame

if __name__ == '__main__':
    args = parser.parse_args()

    if args.image:
        frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
        frame_height, frame_width = frame.shape[:2]
        frame = process_frame(frame, frame_width, frame_height)
        cv2.imwrite('outpy.jpg', frame)
        cv2.imshow("test_window", frame)
        cv2.waitKey(0)

    if args.video:        
        cap = cv2.VideoCapture(args.video)
        if (cap.isOpened() == False): 
            print("Unable to read camera feed")
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps_skip = 1
        fps = int(cap.get(cv2.CAP_PROP_FPS) / fps_skip)
        out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
        # out = cv2.VideoWriter('outpy.mp4',cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width,frame_height))
        frame_id = 0
        while(True):
            ret, frame = cap.read()
            frame_id += 1
            if frame_id % fps_skip:
                continue
            if ret == True:         
                frame = process_frame(frame, frame_width, frame_height)                
                out.write(frame)
            else:
                break 
        cap.release()
        out.release()
