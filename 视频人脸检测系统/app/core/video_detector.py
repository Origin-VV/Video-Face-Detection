from pathlib import Path

import cv2
import torch


class VideoDetector:
    def __init__(self, model_path: str | Path | None = None) -> None:
        self.manual_model_path = Path(model_path) if model_path else None
        self.model_path = Path()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.video_tracker = self.resolve_tracker_path()
        self.model = None
        self.refresh_model_path()

    def build_output_path(self, video_path: str | Path) -> Path:
        source_path = Path(video_path)
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{source_path.stem}_detected.avi"

    def build_image_output_path(self, image_path: str | Path) -> Path:
        source_path = Path(image_path)
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = source_path.suffix.lower() if source_path.suffix else ".jpg"
        if suffix not in {".jpg", ".jpeg", ".png", ".bmp"}:
            suffix = ".jpg"
        return output_dir / f"{source_path.stem}_detected{suffix}"

    # 视频检测
    def detect_video(
        self,
        video_path: str | Path,
        output_path: str | Path | None = None,
        progress_callback=None,
    ) -> dict:
        self.refresh_model_path()
        if not self.is_ready():
            raise FileNotFoundError("未找到可用模型权重。")

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError("视频打开失败。")

        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()

        if width <= 0 or height <= 0:
            raise RuntimeError("视频尺寸无效。")

        output_path = Path(output_path) if output_path else self.build_output_path(video_path)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps if fps and fps > 1 else 25.0,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("结果视频创建失败。")

        model = self.get_model()
        self.reset_trackers()
        processed_frames = 0
        tracked_ids: set[int] = set()
        max_active_tracks = 0

        try:
            track_results = model.track(
                source=str(video_path),
                stream=True,
                persist=False,
                tracker=self.video_tracker,
                conf=0.45,
                device=self.device,
                verbose=False,
            )
            # 绘制框
            for result in track_results:
                annotated_frame = result.plot()
                boxes = getattr(result, "boxes", None)
                track_ids = getattr(boxes, "id", None) if boxes is not None else None
                if track_ids is not None:
                    active_track_ids = track_ids.int().cpu().tolist()
                    tracked_ids.update(active_track_ids)
                    max_active_tracks = max(max_active_tracks, len(active_track_ids))

                writer.write(annotated_frame)

                processed_frames += 1
                if progress_callback is not None:
                    progress_callback(processed_frames, total_frames)
        finally:
            writer.release()

        return {
            "output_path": output_path,
            "processed_frames": processed_frames,
            "total_frames": total_frames,
            "using_fallback_model": self.using_fallback_model(),
            "tracking_enabled": True,
            "tracker_name": self.tracker_display_name(),
            "unique_track_count": len(tracked_ids),
            "max_active_track_count": max_active_tracks,
        }

    # 图片检测
    def detect_image(
        self,
        image_path: str | Path,
        output_path: str | Path | None = None,
    ) -> dict:
        self.refresh_model_path()
        if not self.is_ready():
            raise FileNotFoundError("未找到可用模型权重。")

        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError("图片打开失败。")

        model = self.get_model()
        results = model.predict(
            source=image,
            conf=0.45,
            device=self.device,
            verbose=False,
        )
        annotated_image = results[0].plot()

        output_path = Path(output_path) if output_path else self.build_image_output_path(image_path)
        success = cv2.imwrite(str(output_path), annotated_image)
        if not success:
            raise RuntimeError("检测结果图片保存失败。")

        boxes = getattr(results[0], "boxes", None)
        detection_count = len(boxes) if boxes is not None else 0

        return {
            "output_path": output_path,
            "detection_count": detection_count,
            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),
            "using_fallback_model": self.using_fallback_model(),
        }

    # 摄像头检测
    def detect_frame(self, frame) -> dict:
        self.refresh_model_path()
        if not self.is_ready():
            raise FileNotFoundError("未找到可用模型权重。")

        model = self.get_model()
        results = model.predict(
            source=frame,
            conf=0.45,
            device=self.device,
            verbose=False,
        )
        annotated_frame = results[0].plot()
        boxes = getattr(results[0], "boxes", None)
        detection_count = len(boxes) if boxes is not None else 0

        return {
            "frame": annotated_frame,
            "detection_count": detection_count,
            "using_fallback_model": self.using_fallback_model(),
        }

    def get_model(self):
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))

        return self.model


    # 查找模型
    def resolve_model_path(self) -> Path:
        candidates = [
            Path("models/widerface_yolo11s/weights/best.pt"),
            Path("models/best.pt"),
            Path("runs/detect/models/widerface_yolo11s/weights/best.pt"),
            Path("runs/detect/models/widerface_yolo11s_smoke/weights/best.pt"),
        ]
        candidates.extend(sorted(Path("models").glob("**/weights/best.pt")))
        candidates.extend(sorted(Path("runs/detect/models").glob("**/weights/best.pt")))

        for candidate in candidates:
            if candidate.exists():
                return candidate

        fallback_model = Path("yolo11s.pt")
        if fallback_model.exists():
            return fallback_model

        return candidates[0]

    # 重置追踪器
    def reset_trackers(self) -> None:
        if self.model is None:
            return

        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None)
        if not trackers:
            return

        for tracker in trackers:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()

        if hasattr(predictor, "vid_path"):
            predictor.vid_path = [None] * len(trackers)

    def resolve_tracker_path(self) -> str:
        return str(Path(__file__).resolve().parents[1] / "config" / "bytetrack.yaml")

    def refresh_model_path(self) -> Path:
        next_path = self.manual_model_path or self.resolve_model_path()
        if next_path != self.model_path:
            self.model_path = next_path
            self.model = None
        return self.model_path

    def is_ready(self) -> bool:
        return self.model_path.exists()

    def using_fallback_model(self) -> bool:
        return self.model_path.name == "yolo11s.pt"

    def tracker_display_name(self) -> str:
        return "ByteTrack"

    def model_status_text(self) -> str:
        if not self.is_ready():
            return "未找到可用模型权重。"
        if self.using_fallback_model():
            return "当前使用基础 yolo11s.pt，仅用于流程验证。"
        return f"当前使用人脸权重：{self.model_path.as_posix()}"