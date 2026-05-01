class MotionProfile:
    def __init__(
        self,
        name,
        motion_threshold,
        min_solidity,
        min_area_ratio,
        yolo_conf,
        motion_confidence_min,
        solidity_boost=0.0,
        area_boost=1.0,
    ):
        self.name = name

        # Motion detection
        self.motion_threshold = motion_threshold
        self.min_solidity = min_solidity
        self.min_area_ratio = min_area_ratio

        # YOLO
        self.yolo_conf = yolo_conf
        self.motion_confidence_min = motion_confidence_min

        # Optional dynamic tuning
        self.solidity_boost = solidity_boost
        self.area_boost = area_boost