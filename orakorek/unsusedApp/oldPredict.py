import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
import supervision as sv
import time
import os
from concurrent.futures import ThreadPoolExecutor
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SOURCE Points for Perspective Transformation
SOURCE = np.array([[-5500, 2500], [3800, 2500], [1400, 300], [900, 320]])  # Adjust according to your needs
TARGET_WIDTH = 25
TARGET_HEIGHT = 250

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

model = YOLO("models/vehicle-accident.pt").to(device)

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def draw_text_with_background(image, text, position, font, scale, color, thickness, background_color):
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    cv2.rectangle(image, (x, y - text_height - baseline), (x + text_width, y + baseline), background_color, -1)
    cv2.putText(image, text, (x, y), font, scale, color, thickness)

if __name__ == "__main__":
    source_video_path = "dataTest/1125.mp4"  # Adjust path as necessary
    target_video_path = "outputTest/output_accident.mp4"  # Adjust path as necessary
    confidence_threshold = 0.6
    iou_threshold = 0.5
    cap = cv2.VideoCapture(source_video_path)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    
    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps, track_activation_threshold=confidence_threshold
    )

    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    bounding_box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER,
    )

    def read_frames(batch_size=64):
        frames = []
        while True:
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if frames:
                yield frames
                frames = []
            else:
                break

    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
    vehicle_counts_per_second = defaultdict(lambda: defaultdict(set))

    pixels_to_meter = 0.2  # Adjust as per your calibration

    car_speeds = []
    truck_speeds = []
    motorcycle_speeds = []
    bus_speeds = []  # Add bus speed tracking
    total_vehicles = defaultdict(int)

    with sv.VideoSink(target_video_path, video_info) as sink, ThreadPoolExecutor(max_workers=12) as executor:
        for batch_frames in read_frames():
            processed_frames = list(executor.map(preprocess_frame, batch_frames))

            for original_frame, frame in zip(batch_frames, processed_frames):
                original_frame = original_frame.copy()
                result = model(frame)[0]
                detections = sv.Detections.from_ultralytics(result)

                detections = detections[detections.confidence > confidence_threshold]

                detections = detections[polygon_zone.trigger(detections)]
                detections = detections.with_nms(threshold=iou_threshold)
                detections = byte_track.update_with_detections(detections=detections)

                points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                points = view_transformer.transform_points(points=points).astype(int)

                labels = []
                current_second = int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)

                total_vehicles['car'] += len(vehicle_counts_per_second[current_second]['car'] - set(total_vehicles.get('car_ids', set())))
                total_vehicles['truck'] += len(vehicle_counts_per_second[current_second]['truck'] - set(total_vehicles.get('truck_ids', set())))
                total_vehicles['motorcycle'] += len(vehicle_counts_per_second[current_second]['motorcycle'] - set(total_vehicles.get('motorcycle_ids', set())))
                total_vehicles['bus'] += len(vehicle_counts_per_second[current_second]['bus'] - set(total_vehicles.get('bus_ids', set())))

                total_vehicles['car_ids'] = total_vehicles.get('car_ids', set()) | vehicle_counts_per_second[current_second]['car']
                total_vehicles['truck_ids'] = total_vehicles.get('truck_ids', set()) | vehicle_counts_per_second[current_second]['truck']
                total_vehicles['motorcycle_ids'] = total_vehicles.get('motorcycle_ids', set()) | vehicle_counts_per_second[current_second]['motorcycle']
                total_vehicles['bus_ids'] = total_vehicles.get('bus_ids', set()) | vehicle_counts_per_second[current_second]['bus']

                for i, (class_id, tracker_id) in enumerate(zip(detections.class_id, detections.tracker_id)):
                    if class_id == 1:  # car
                        vehicle_counts_per_second[current_second]['car'].add(tracker_id)
                    elif class_id == 6:  # motorcycle
                        vehicle_counts_per_second[current_second]['motorcycle'].add(tracker_id)
                    elif class_id == 10:  # truck
                        vehicle_counts_per_second[current_second]['truck'].add(tracker_id)
                    elif class_id == 0:  # bus
                        vehicle_counts_per_second[current_second]['bus'].add(tracker_id)

                    y = points[i][1]
                    coordinates[tracker_id].append(y)
                    if len(coordinates[tracker_id]) < video_info.fps / 2:
                        labels.append(f"#{tracker_id}")
                    else:
                        coordinate_start = coordinates[tracker_id][-1]
                        coordinate_end = coordinates[tracker_id][0]
                        distance_in_pixels = abs(coordinate_start - coordinate_end)

                        distance_in_meters = distance_in_pixels * pixels_to_meter
                        time = len(coordinates[tracker_id]) / video_info.fps
                        speed_m_per_s = distance_in_meters / time
                        speed_km_per_h = speed_m_per_s * 3.6

                        if class_id == 1:
                            car_speeds.append((tracker_id, speed_km_per_h))
                        elif class_id == 10:
                            truck_speeds.append((tracker_id, speed_km_per_h))
                        elif class_id == 6:
                            motorcycle_speeds.append((tracker_id, speed_km_per_h))
                        elif class_id == 0:
                            bus_speeds.append((tracker_id, speed_km_per_h))

                        labels.append(f"#{tracker_id} {int(speed_km_per_h)} km/h")

                annotated_frame = original_frame.copy()
                try:
                    if len(detections) == len(detections.tracker_id):
                        annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
                except IndexError as e:
                    print(f"IndexError occurred: {e}")

                annotated_frame = bounding_box_annotator.annotate(scene=annotated_frame, detections=detections)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

                cv2.polylines(annotated_frame, [SOURCE.astype(np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)
                
                total_vehicles['car'] += len(vehicle_counts_per_second[current_second]['car'] - set(total_vehicles.get('car_ids', set())))
                total_vehicles['truck'] += len(vehicle_counts_per_second[current_second]['truck'] - set(total_vehicles.get('truck_ids', set())))
                total_vehicles['motorcycle'] += len(vehicle_counts_per_second[current_second]['motorcycle'] - set(total_vehicles.get('motorcycle_ids', set())))
                total_vehicles['bus'] += len(vehicle_counts_per_second[current_second]['bus'] - set(total_vehicles.get('bus_ids', set())))
                
                total_vehicles['car_ids'] = total_vehicles.get('car_ids', set()) | vehicle_counts_per_second[current_second]['car']
                total_vehicles['truck_ids'] = total_vehicles.get('truck_ids', set()) | vehicle_counts_per_second[current_second]['truck']
                total_vehicles['motorcycle_ids'] = total_vehicles.get('motorcycle_ids', set()) | vehicle_counts_per_second[current_second]['motorcycle']
                total_vehicles['bus_ids'] = total_vehicles.get('bus_ids', set()) | vehicle_counts_per_second[current_second]['bus']

                car_text = f"Cars: {total_vehicles['car']}"
                truck_text = f"Trucks: {total_vehicles['truck']}"
                motorcycle_text = f"Motorcycles: {total_vehicles['motorcycle']}"
                bus_text = f"Buses: {total_vehicles['bus']}"

                draw_text_with_background(annotated_frame, car_text, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), thickness, (255, 255, 255))
                draw_text_with_background(annotated_frame, truck_text, (30, 160), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), thickness, (255, 255, 255))
                draw_text_with_background(annotated_frame, motorcycle_text, (30, 230), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 0, 0), thickness, (255, 255, 255))
                draw_text_with_background(annotated_frame, bus_text, (30, 300), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 255), thickness, (255, 255, 255))

                sink.write_frame(annotated_frame)
                cv2.imshow("frame", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cv2.destroyAllWindows()

    # Calculating and writing speed statistics to a text file
    if car_speeds:
        avg_car_speed = np.mean([speed for _, speed in car_speeds])
        fastest_car = max(car_speeds, key=lambda x: x[1])
    else:
        avg_car_speed = 0
        fastest_car = (0, 0)

    if truck_speeds:
        avg_truck_speed = np.mean([speed for _, speed in truck_speeds])
        fastest_truck = max(truck_speeds, key=lambda x: x[1])
    else:
        avg_truck_speed = 0
        fastest_truck = (0, 0)

    if motorcycle_speeds:
        avg_motorcycle_speed = np.mean([speed for _, speed in motorcycle_speeds])
        fastest_motorcycle = max(motorcycle_speeds, key=lambda x: x[1])
    else:
        avg_motorcycle_speed = 0
        fastest_motorcycle = (0, 0)

    if bus_speeds:
        avg_bus_speed = np.mean([speed for _, speed in bus_speeds])
        fastest_bus = max(bus_speeds, key=lambda x: x[1])
    else:
        avg_bus_speed = 0
        fastest_bus = (0, 0)

    output_file_path = os.path.abspath('hasil.txt')
    output_dir = os.path.dirname(output_file_path)
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(output_file_path, 'w') as f:
            f.write(f"Rata-rata kecepatan mobil = {avg_car_speed:.2f} km/h\n")
            f.write(f"Mobil tercepat adalah Car #{fastest_car[0]} dengan kecepatan {fastest_car[1]:.2f} km/h\n\n")

            f.write(f"Rata-rata kecepatan motor = {avg_motorcycle_speed:.2f} km/h\n")
            f.write(f"Motor tercepat adalah Motorcycle #{fastest_motorcycle[0]} dengan kecepatan {fastest_motorcycle[1]:.2f} km/h\n\n")

            f.write(f"Rata-rata kecepatan truk = {avg_truck_speed:.2f} km/h\n")
            f.write(f"Truk tercepat adalah Truck #{fastest_truck[0]} dengan kecepatan {fastest_truck[1]:.2f} km/h\n\n")

            f.write(f"Rata-rata kecepatan bus = {avg_bus_speed:.2f} km/h\n")
            f.write(f"Bus tercepat adalah Bus #{fastest_bus[0]} dengan kecepatan {fastest_bus[1]:.2f} km/h\n\n")

            f.write("Keterangan tiap detik:\n")
            for second, counts in vehicle_counts_per_second.items():
                car_count = len(counts['car'])
                motorcycle_count = len(counts['motorcycle'])
                truck_count = len(counts['truck'])
                bus_count = len(counts['bus'])
                line = f"Detik {second}: Mobil={car_count}, Motor={motorcycle_count}, Truk={truck_count}, Bus={bus_count}\n"
                f.write(line)
    except Exception as e:
        print(f"Error saat mencoba menulis ke file: {e}")
    else:
        print(f"Data berhasil disimpan di: {output_file_path}")
