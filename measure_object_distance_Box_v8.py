from realsense_camera import *
import cv2
import argparse

from ultralytics import YOLO
import numpy as np
import supervision as sv

rs = RealsenseCamera()

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    model = YOLO("yolov8s-seg.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )
    

    while True:
        ret, frame, depth_frame = rs.get_frame_stream()
        
        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
        
        object_frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
       )
        
        coord = detections.xyxy
        
        for i in range(len(coord)):
            x1 = coord[i][0]
            x2 = coord[i][2]
            y1 = coord[i][1]
            y2 = coord[i][3]
            center_x = (x1+x2)/2
            center_y = (y1+y2)/2
            width = x2-x1
            height = y2-y1
            print("Center coordinate is",(center_x,center_y))
 
        # depth_frame = np.array(depth_frame)
        depth_frame2 = depth_frame.copy()
        
        center_y = int(center_y)
        center_x = int(center_x)
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        width = int(width)
        height = int(height)
        
        sum_values = 0
        
        
        for i in range(x1,x2+1):
            for j in range(y1,y2+1):
                sum_values += depth_frame[[j],[i]]
                depth_frame2[[j],[i]]=1000
        print(y1,y2)
         
        depth_mm = sum_values / (width*height)
        print("Depth in the center is", depth_mm/10,"cm")
        
        
        cv2.imshow("depth2",depth_frame2)
        cv2.imshow("depth",depth_frame)
        cv2.imshow("color", frame)
        

        if (cv2.waitKey(30) == 27):
            break


if __name__ == "__main__":
    main()

rs.release()
cv2.destroyAllWindows()
