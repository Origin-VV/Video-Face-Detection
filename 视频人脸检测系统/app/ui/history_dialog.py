import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout


class HistoryVideoDialog(QDialog):
    def __init__(self, video_path: str, parent=None) -> None:
        super().__init__(parent)
        self.video_path = video_path
        self.capture = None
        self.frame_interval_ms = 40
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._play_next_frame)

        self.setWindowTitle("历史检测视频预览")
        self.resize(960, 720)
        self._build_ui()
        self._apply_styles()
        self._open_video()

    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self.path_label = QLabel(self.video_path)
        self.path_label.setWordWrap(True)
        self.path_label.setObjectName("pathLabel")

        self.video_label = QLabel("正在加载历史检测视频...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(860, 560)
        self.video_label.setObjectName("videoPreview")

        self.preview_slider = QSlider(Qt.Horizontal)
        self.preview_slider.setRange(1, 1)
        self.preview_slider.setValue(1)
        self.preview_slider.setEnabled(False)
        self.preview_slider.sliderPressed.connect(self._handle_slider_pressed)
        self.preview_slider.sliderMoved.connect(self._handle_slider_moved)
        self.preview_slider.sliderReleased.connect(self._handle_slider_released)

        button_row = QHBoxLayout()
        self.play_button = QPushButton("播放视频")
        self.close_button = QPushButton("关闭")
        self.play_button.setObjectName("primaryButton")
        self.close_button.setObjectName("secondaryButton")
        self.play_button.clicked.connect(self._toggle_playback)
        self.close_button.clicked.connect(self.accept)
        self.play_button.setEnabled(False)

        button_row.addWidget(self.play_button)
        button_row.addStretch()
        button_row.addWidget(self.close_button)

        layout.addWidget(self.path_label)
        layout.addWidget(self.video_label, 1)
        layout.addWidget(self.preview_slider)
        layout.addLayout(button_row)
        self.setLayout(layout)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background-color: #f4f8fb;
                color: #223247;
                font-size: 13px;
            }
            QLabel#pathLabel {
                color: #5f7387;
            }
            QLabel#videoPreview {
                border: 2px dashed #b8c8d8;
                border-radius: 18px;
                background-color: #111827;
                color: white;
            }
            QPushButton {
                min-height: 38px;
                border-radius: 10px;
                font-weight: 600;
                padding: 0 14px;
            }
            QPushButton#primaryButton {
                background-color: #1f6feb;
                color: white;
                border: none;
            }
            QPushButton#secondaryButton {
                background-color: #ffffff;
                color: #1f3a56;
                border: 1px solid #c8d5e2;
            }
            """
        )

    # 打开视频
    def _open_video(self) -> None:
        self._release_capture()
        self.capture = cv2.VideoCapture(self.video_path)
        if not self.capture.isOpened():
            self.video_label.setText("历史视频打开失败。")
            return

        fps = self.capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_interval_ms = max(1, int(1000 / fps)) if fps and fps > 1 else 40
        if not self._display_frame_at(0):
            self.video_label.setText("历史视频读取失败。")
            self._release_capture()
            return

        self.preview_slider.setRange(1, max(1, total_frames))
        self.preview_slider.setEnabled(True)
        self.play_button.setEnabled(True)

    # 播放/暂停
    def _toggle_playback(self) -> None:
        if self.capture is None:
            return

        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("播放视频")
            return

        self.timer.start(self.frame_interval_ms)
        self.play_button.setText("暂停播放")

    # 渲染下一帧
    def _play_next_frame(self) -> None:
        if self.capture is None:
            return

        ok, frame = self.capture.read()
        if ok:
            self._show_frame(frame)
            self._update_slider_position()
            return

        self.timer.stop()
        self.play_button.setText("重新播放")
        self._display_frame_at(0)

    # 拖拽开始
    def _handle_slider_pressed(self) -> None:
        if self.capture is None:
            return

        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("播放视频")

    # 拖拽移动
    def _handle_slider_moved(self, slider_value: int) -> None:
        if self.capture is None:
            return

        self._display_frame_at(slider_value - 1)

    # 拖拽结束
    def _handle_slider_released(self) -> None:
        if self.capture is None:
            return

        self._display_frame_at(self.preview_slider.value() - 1)

    # 通道转换
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

    # 定点跳转
    def _display_frame_at(self, frame_index: int) -> bool:
        if self.capture is None:
            return False

        total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return False

        target_frame_index = max(0, min(frame_index, total_frames - 1))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
        ok, frame = self.capture.read()
        if not ok:
            return False

        self._show_frame(frame)
        self.preview_slider.setValue(target_frame_index + 1)
        return True

    # 同步进度条
    def _update_slider_position(self) -> None:
        if self.capture is None:
            return

        current_position = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
        self.preview_slider.setValue(max(1, current_position))

    # 释放资源
    def _release_capture(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.preview_slider.setRange(1, 1)
        self.preview_slider.setValue(1)
        self.preview_slider.setEnabled(False)

    # 退出拦截
    def closeEvent(self, event) -> None:
        self._release_capture()
        super().closeEvent(event)