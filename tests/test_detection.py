import numpy as np
import pytest

from src.detection.model import Detection, DetectionModel, VEHICLE_CLASS_IDS
from src.detection.detector import VehicleDetector, FrameAnalysis


def test_detection_center():
    det = Detection(bbox=(10, 20, 110, 70), confidence=0.9, class_id=2, class_name="car")
    assert det.center == (60.0, 45.0)


def test_detection_area():
    det = Detection(bbox=(0, 0, 100, 50), confidence=0.8, class_id=2, class_name="car")
    assert det.area == 5000.0


def test_detection_dimensions():
    det = Detection(bbox=(10, 10, 60, 40), confidence=0.7, class_id=5, class_name="bus")
    assert det.width == 50.0
    assert det.height == 30.0


class MockDetectionModel:

    def predict(self, frame: np.ndarray):
        return [
            Detection((50, 50, 150, 120), 0.95, 2, "car"),
            Detection((200, 80, 320, 180), 0.88, 2, "car"),
            Detection((400, 60, 520, 160), 0.75, 7, "truck"),
        ]


@pytest.fixture
def detector():
    return VehicleDetector(
        model=MockDetectionModel(),
        frame_width=640,
        frame_height=480,
        num_lanes=4,
    )


def test_process_frame_count(detector, sample_frame):
    analysis = detector.process_frame(sample_frame, frame_id=0)
    assert isinstance(analysis, FrameAnalysis)
    assert analysis.vehicle_count == 3


def test_process_frame_density_positive(detector, sample_frame):
    analysis = detector.process_frame(sample_frame)
    assert analysis.density >= 0.0


def test_lane_counts_sum(detector, sample_frame):
    analysis = detector.process_frame(sample_frame)
    total = sum(analysis.lane_counts.values())
    assert total == analysis.vehicle_count


def test_heatmap_shape(detector, sample_frame):
    analysis = detector.process_frame(sample_frame)
    assert analysis.heatmap.shape == (480, 640)


def test_roi_filter():
    det = VehicleDetector(
        model=MockDetectionModel(),
        frame_width=640,
        frame_height=480,
        roi=(0, 0, 100, 100),
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    analysis = det.process_frame(frame)
    assert analysis.vehicle_count <= 1


def test_queue_length_empty(detector):
    q = detector.compute_queue_length([], stop_line_y=300)
    assert q == 0.0


def test_queue_length_nonzero(detector):
    dets = [Detection((100, 200, 200, 280), 0.9, 2, "car")]
    q = detector.compute_queue_length(dets, stop_line_y=300, frame_height=480)
    assert q > 0.0
