from time import perf_counter
from typing import Callable
import winsound

import torch
import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core.video_detector import VideoDetector


class CameraDetectionPage(QWidget):
    def __init__(
        self,
        current_user: str,
        status_callback: Callable[[str], None] | None = None,
        data_changed_callback: Callable[[], None] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.current_user = current_user
        self.status_callback = status_callback
        self.data_changed_callback = data_changed_callback

        self.capture = None
        self.camera_index = 0
        self.frame_count = 0
        self.last_fps_time = perf_counter()
        self.last_fps_frame_count = 0
        self.detector = VideoDetector()
        self.alarm_enabled = False
        self.alarm_cooldown_seconds = 3
        self.last_alarm_time = 0.0

        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self._process_next_frame)

        self._build_ui()
        self._refresh_model_info()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(14)

        title_card = QFrame()
        title_card.setObjectName("moduleHeroCard")
        title_layout = QVBoxLayout()
        title_layout.setContentsMargins(18, 18, 18, 18)
        title_layout.setSpacing(6)

        title = QLabel("摄像头检测")
        title.setObjectName("moduleTitle")
        title_layout.addWidget(title)
        title_card.setLayout(title_layout)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(14)
        content_layout.addWidget(self._build_left_panel(), 3)
        content_layout.addWidget(self._build_center_panel(), 7)
        content_layout.addWidget(self._build_right_panel(), 3)

        root_layout.addWidget(title_card)
        root_layout.addLayout(content_layout)
        root_layout.addWidget(self._build_footer())
        self.setLayout(root_layout)

    def _build_left_panel(self) -> QWidget:
        panel = QVBoxLayout()
        panel.setSpacing(14)

        control_box = QGroupBox("操作区")
        control_layout = QVBoxLayout()

        self.start_button = QPushButton("打开摄像头")
        self.stop_button = QPushButton("关闭摄像头")
        self.alarm_button = QPushButton("打开警报")

        self.start_button.setObjectName("primaryButton")
        self.stop_button.setObjectName("secondaryButton")
        self.alarm_button.setObjectName("secondaryButton")
        self.stop_button.setEnabled(False)

        self.start_button.clicked.connect(self._start_camera)
        self.stop_button.clicked.connect(self._stop_camera)
        self.alarm_button.clicked.connect(self._toggle_alarm)

        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.alarm_button)
        control_box.setLayout(control_layout)

        info_box = QGroupBox("摄像头信息")
        info_layout = QGridLayout()
        info_layout.setHorizontalSpacing(10)
        info_layout.setVerticalSpacing(8)
        info_layout.setColumnMinimumWidth(0, 76)
        info_layout.setColumnStretch(1, 1)

        self.camera_status_label = QLabel("未打开")
        self.model_status_label = QLabel("--")
        self.resolution_label = QLabel("--")
        self.fps_label = QLabel("--")
        self.frame_count_label = QLabel("0")
        self.face_count_label = QLabel("0")
        self.alarm_status_label = QLabel("未触发")
        self.alarm_status_label.setObjectName("alarmStatusNormal")
        self.alarm_status_label.setAlignment(Qt.AlignCenter)
        self.alarm_status_label.setMinimumHeight(42)

        self.model_status_label.setWordWrap(True)

        info_layout.addWidget(QLabel("摄像头"), 0, 0)
        info_layout.addWidget(self.camera_status_label, 0, 1)
        info_layout.addWidget(QLabel("画面尺寸"), 1, 0)
        info_layout.addWidget(self.resolution_label, 1, 1)
        info_layout.addWidget(QLabel("实时帧率"), 2, 0)
        info_layout.addWidget(self.fps_label, 2, 1)
        info_layout.addWidget(QLabel("处理帧数"), 3, 0)
        info_layout.addWidget(self.frame_count_label, 3, 1)
        info_layout.addWidget(QLabel("检测人脸"), 4, 0)
        info_layout.addWidget(self.face_count_label, 4, 1)
        info_layout.addWidget(QLabel("警报状态"), 5, 0)
        info_layout.addWidget(self.alarm_status_label, 5, 1)
        info_box.setLayout(info_layout)

        guide_box = QGroupBox("操作提示")
        guide_layout = QVBoxLayout()
        guide_text = QLabel(
            "1. 点击打开摄像头\n"
            "2. 系统会实时检测画面中的人脸\n"
            "3. 离开页面或关闭窗口会自动释放摄像头"
        )
        guide_text.setWordWrap(True)
        guide_layout.addWidget(guide_text)
        guide_box.setLayout(guide_layout)

        panel.addWidget(control_box)
        panel.addWidget(info_box)
        panel.addWidget(guide_box)
        panel.addStretch()

        container = QWidget()
        container.setLayout(panel)
        return container

    # 预览面板
    def _build_center_panel(self) -> QWidget:
        center = QFrame()
        center.setObjectName("previewCard")
        layout = QVBoxLayout()
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        preview_title = QLabel("摄像头实时预览")
        preview_title.setObjectName("sectionTitle")

        self.video_label = QLabel("请先打开摄像头")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(760, 560)
        self.video_label.setObjectName("videoPreview")

        layout.addWidget(preview_title)
        layout.addWidget(self.video_label, 1)
        center.setLayout(layout)
        return center

    # 右侧面板
    def _build_right_panel(self) -> QWidget:
        panel = QVBoxLayout()
        panel.setSpacing(12)

        log_box = QGroupBox("运行日志")
        log_layout = QVBoxLayout()
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        log_layout.addWidget(self.log_output)
        log_box.setLayout(log_layout)

        panel.addWidget(log_box, 1)
        panel.addStretch()

        container = QWidget()
        container.setLayout(panel)
        return container

    # 底部状态栏
    def _build_footer(self) -> QWidget:
        footer = QFrame()
        footer.setObjectName("footerCard")
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        self.status_label = QLabel("当前状态：等待打开摄像头。")
        self.status_label.setObjectName("statusText")

        layout.addWidget(self.status_label)
        footer.setLayout(layout)
        return footer

    # 启动摄像头
    def _start_camera(self) -> None:
        self._refresh_model_info()
        if not self.detector.is_ready():
            self._append_log("未找到可用模型权重。")
            self._push_status("当前状态：未找到可用模型权重。")
            return

        self._release_capture()
        # 尝试使用 DirectShow 模式打开提高兼容性
        self.capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = cv2.VideoCapture(self.camera_index)

        if not self.capture.isOpened():
            self.capture = None
            self._append_log("摄像头打开失败，请确认摄像头可用且未被其他程序占用。")
            self._push_status("当前状态：摄像头打开失败。")
            return

        self.frame_count = 0
        self.last_fps_time = perf_counter()
        self.last_fps_frame_count = 0
        self.last_alarm_time = 0.0
        self.alarm_status_label.setText("未触发")
        self.camera_status_label.setText(f"已打开：{self.camera_index}")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.timer.start()

        if self.detector.using_fallback_model():
            self._append_log("当前使用基础 yolo11s.pt，仅用于流程验证。")
        else:
            self._append_log(f"摄像头检测已启动，使用权重：{self.detector.model_path.as_posix()}")
        self._push_status("当前状态：摄像头实时检测中。")

    # 停止摄像头
    def _stop_camera(self) -> None:
        self._release_capture()
        self._push_status("当前状态：摄像头已关闭。")
        self._append_log("摄像头已关闭。")

    # 处理下一帧
    def _process_next_frame(self) -> None:
        if self.capture is None:
            return

        ok, frame = self.capture.read()
        if not ok:
            self._append_log("摄像头画面读取失败。")
            self._push_status("当前状态：摄像头画面读取失败。")
            self._release_capture()
            return

        try:
            # 实时检测单帧画面
            result = self.detector.detect_frame(frame)
        except Exception as exc:
            self._append_log(f"摄像头检测失败：{exc}")
            self._push_status(f"当前状态：摄像头检测失败，原因：{exc}")
            self._release_capture()
            return

        annotated_frame = result["frame"]
        self.frame_count += 1
        self._handle_face_alarm(result["detection_count"])
        self._show_frame(annotated_frame)
        self._update_runtime_info(annotated_frame, result["detection_count"])

    # 警报开关
    def _toggle_alarm(self) -> None:
        self.alarm_enabled = not self.alarm_enabled
        if self.alarm_enabled:
            self.alarm_button.setText("关闭警报")
            self._set_alarm_status("已开启", alarm=False)
            self._append_log("人脸警报已开启。")
            return

        self.alarm_button.setText("打开警报")
        self._set_alarm_status("已关闭", alarm=False)
        self._append_log("人脸警报已关闭。")

    # 人脸警报
    def _handle_face_alarm(self, detection_count: int) -> None:
        if detection_count <= 0:
            self._set_alarm_status("已开启" if self.alarm_enabled else "已关闭", alarm=False)
            return
        if not self.alarm_enabled:
            self._set_alarm_status("已检测，警报关闭", alarm=False)
            return

        now = perf_counter()
        if now - self.last_alarm_time < self.alarm_cooldown_seconds:
            self._set_alarm_status("警报中：检测到人脸", alarm=True)
            return

        self.last_alarm_time = now
        self._set_alarm_status("警报中：检测到人脸", alarm=True)
        self._append_log(f"警报：检测到 {detection_count} 张人脸。")
        self._push_status(f"当前状态：警报，检测到 {detection_count} 张人脸。")
        self._play_alarm_sound()

    def _set_alarm_status(self, text: str, alarm: bool) -> None:
        self.alarm_status_label.setText(text)
        self.alarm_status_label.setObjectName("alarmStatusAlert" if alarm else "alarmStatusNormal")
        self.alarm_status_label.style().unpolish(self.alarm_status_label)
        self.alarm_status_label.style().polish(self.alarm_status_label)
        self.alarm_status_label.update()

    def _play_alarm_sound(self) -> None:
        try:
            for frequency in (1000, 1400, 1000):
                winsound.Beep(frequency, 180)
        except RuntimeError:
            QApplication.beep()

    # 更新信息
    def _update_runtime_info(self, frame, detection_count: int) -> None:
        height, width = frame.shape[:2]
        self.resolution_label.setText(f"{width} x {height}")
        self.frame_count_label.setText(str(self.frame_count))
        self.face_count_label.setText(str(detection_count))

        # 计算实时帧率 FPS
        now = perf_counter()
        elapsed = now - self.last_fps_time
        if elapsed >= 1:
            processed = self.frame_count - self.last_fps_frame_count
            self.fps_label.setText(f"{processed / elapsed:.1f}")
            self.last_fps_time = now
            self.last_fps_frame_count = self.frame_count

    # 显示画面
    def _show_frame(self, frame) -> None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        bytes_per_line = channels * width
        image = QImage(
            rgb_frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    # 刷新模型
    def _refresh_model_info(self) -> None:
        self.detector.refresh_model_path()
        self.model_status_label.setText(self.detector.model_status_text())

    # 记录日志
    def _append_log(self, text: str) -> None:
        self.log_output.appendPlainText(text)

    # 推送状态
    def _push_status(self, text: str) -> None:
        self.status_label.setText(text)
        if self.status_callback is not None:
            self.status_callback(text)

    # 页面隐藏
    def on_page_hidden(self) -> None:
        self._release_capture()

    # 释放资源
    def release_resources(self) -> None:
        self._release_capture()

    # 停止采集
    def _release_capture(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
        if self.capture is not None:
            self.capture.release()
            self.capture = None

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.camera_status_label.setText("未打开")
