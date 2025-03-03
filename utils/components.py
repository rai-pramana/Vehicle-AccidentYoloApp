import torch
from collections import deque, defaultdict
from ultralytics import YOLO
import supervision as sv
from utils.viewTransformer import ViewTransformer

def load_model_and_initialize_components(model_path, selected_model, resolution_wh, confidence_threshold, SOURCE, TARGET, FPS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(f'{model_path}/{selected_model}').to(device)

    all_classes = model.names.values()
    vehicle_classes = {"bus", "car", "motorcycle", "truck"}
    accident_classes = set(all_classes) - vehicle_classes

    byte_track = sv.ByteTrack(
        frame_rate=FPS, 
        track_activation_threshold=confidence_threshold
    )

    thickness = int(sv.calculate_optimal_line_thickness(resolution_wh) / 2)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh)

    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=FPS * 2,
        position=sv.Position.BOTTOM_CENTER,
    )

    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
    coordinates = defaultdict(lambda: deque(maxlen=int(FPS)))

    return model, byte_track, box_annotator, label_annotator, trace_annotator, polygon_zone, view_transformer, coordinates, vehicle_classes, accident_classes, thickness, text_scale