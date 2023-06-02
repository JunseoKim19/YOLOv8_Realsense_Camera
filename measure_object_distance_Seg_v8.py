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
        ret, frame, depth_frame, depth_colormap = rs.get_frame_stream()
        
        
#        accel_data, _=rs.get_motion_data()
#        pitch, roll = rs.calculate_pitch_yaw(accel_data)
        
        
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
            
            center = (int(center_x), int(center_y))
            depth_mm = depth_frame[center[::-1]] / 10
            
#            actual_distance = depth_mm / (cos(pitch*pi/180) * cos(roll*pi/180))

#            print(f"Actual distance in the center for {labels[i]} is {actual_distance} cm")
            
            print(f"Depth in the center for {labels[i]} is {depth_mm} cm")
            
            segmentation_contours_idx = []
            for seg in result.masks.segments:
                #contours
                segment = np.array(seg, dtype=np.int32)
                segment[:, 0] = x1 + (seg[:, 0] * width).astype(int)
                segment[:, 1] = y1 + (seg[:, 1] * height).astype(int)
                segmentation_contours_idx.append(segment)
        
  
        cv2.imshow("depth", depth_colormap)
        cv2.imshow("color", frame)

        

        if (cv2.waitKey(30) == 27):
            break


if __name__ == "__main__":
    main()

rs.release()
cv2.destroyAllWindows()