import shutil
from pathlib import Path
from typing import Callable

import torch
import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.core.database import count_detection_records, save_detection_record
from app.core.video_detector import VideoDetector


class VideoDetectionPage(QWidget):
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
        self.source_video_path = ""
        self.preview_video_path = ""
        self.detected_video_path = ""
        self.frame_interval_ms = 40
        self.detector = VideoDetector()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._play_next_frame)

        self._build_ui()
        self._refresh_system_info()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(14)

        title_card = QFrame()
        title_card.setObjectName("moduleHeroCard")
        title_layout = QVBoxLayout()
        title_layout.setContentsMargins(18, 18, 18, 18)
        title_layout.setSpacing(6)

        title = QLabel("视频检测")
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

        self.load_button = QPushButton("加载视频")
        self.play_button = QPushButton("播放视频")
        self.detect_button = QPushButton("开始检测")
        self.export_button = QPushButton("导出结果")

        self.load_button.setObjectName("primaryButton")
        self.play_button.setObjectName("secondaryButton")
        self.detect_button.setObjectName("primaryButton")
        self.export_button.setObjectName("secondaryButton")

        self.play_button.setEnabled(False)
        self.detect_button.setEnabled(False)
        self.export_button.setEnabled(False)

        self.load_button.clicked.connect(self._load_video)
        self.play_button.clicked.connect(self._toggle_playback)
        self.detect_button.clicked.connect(self._run_detection)
        self.export_button.clicked.connect(self._export_result)

        control_layout.addWidget(self.load_button)
        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.detect_button)
        control_layout.addWidget(self.export_button)
        control_box.setLayout(control_layout)

        info_box = QGroupBox("视频信息")
        info_layout = QGridLayout()
        info_layout.setHorizontalSpacing(10)
        info_layout.setVerticalSpacing(8)
        info_layout.setColumnMinimumWidth(0, 76)
        info_layout.setColumnStretch(1, 1)

        self.source_path_label = QLabel("未加载")
        self.result_path_label = QLabel("未生成")
        self.frame_count_label = QLabel("0")
        self.current_frame_label = QLabel("0 / 0")
        self.fps_label = QLabel("--")
        self.record_count_label = QLabel("0")
        self.record_status_label = QLabel("尚未写入")

        for label in [self.source_path_label, self.result_path_label]:
            label.setWordWrap(True)

        info_layout.addWidget(QLabel("原始视频"), 0, 0)
        info_layout.addWidget(self.source_path_label, 0, 1)
        info_layout.addWidget(QLabel("结果视频"), 1, 0)
        info_layout.addWidget(self.result_path_label, 1, 1)
        info_layout.addWidget(QLabel("总帧数"), 2, 0)
        info_layout.addWidget(self.frame_count_label, 2, 1)
        info_layout.addWidget(QLabel("当前进度"), 3, 0)
        info_layout.addWidget(self.current_frame_label, 3, 1)
        info_layout.addWidget(QLabel("视频帧率"), 4, 0)
        info_layout.addWidget(self.fps_label, 4, 1)
        info_layout.addWidget(QLabel("视频记录"), 5, 0)
        info_layout.addWidget(self.record_count_label, 5, 1)
        info_layout.addWidget(QLabel("写入状态"), 6, 0)
        info_layout.addWidget(self.record_status_label, 6, 1)
        info_box.setLayout(info_layout)

        guide_box = QGroupBox("操作提示")
        guide_layout = QVBoxLayout()
        guide_text = QLabel(
            "1. 加载视频\n"
            "2. 开始检测\n"
            "3. 预览并导出结果"
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

    def _build_center_panel(self) -> QWidget:
        center = QFrame()
        center.setObjectName("previewCard")
        layout = QVBoxLayout()
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        preview_title = QLabel("视频预览区")
        preview_title.setObjectName("sectionTitle")

        self.video_label = QLabel("请先加载本地视频")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(760, 560)
        self.video_label.setObjectName("videoPreview")

        self.preview_slider = QSlider(Qt.Horizontal)
        self.preview_slider.setRange(1, 1)
        self.preview_slider.setValue(1)
        self.preview_slider.setEnabled(False)
        self.preview_slider.sliderPressed.connect(self._handle_slider_pressed)
        self.preview_slider.sliderMoved.connect(self._handle_slider_moved)
        self.preview_slider.sliderReleased.connect(self._handle_slider_released)

        layout.addWidget(preview_title)
        layout.addWidget(self.video_label, 1)
        layout.addWidget(self.preview_slider)
        center.setLayout(layout)
        return center

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

    def _build_footer(self) -> QWidget:
        footer = QFrame()
        footer.setObjectName("footerCard")
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        self.status_label = QLabel("当前状态：等待加载视频。")
        self.status_label.setObjectName("statusText")

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")

        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        footer.setLayout(layout)
        return footer

    # 文件加载
    def _load_video(self) -> None:
        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )
        if not video_path:
            return

        if not self._open_video_for_preview(video_path):
            self._push_status("当前状态：视频打开失败。")
            self._append_log(f"打开视频失败：{video_path}")
            return

        self.source_video_path = video_path
        self.preview_video_path = video_path
        self.detected_video_path = ""
        self.export_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.record_status_label.setText("尚未写入")

        self.detector.refresh_model_path()
        self.detect_button.setEnabled(self.detector.is_ready())
        self.play_button.setEnabled(True)
        self.play_button.setText("播放视频")

        self._refresh_system_info()
        self._push_status("当前状态：已加载视频，等待开始检测。")
        self._append_log(f"已加载视频：{video_path}")

    # 播放与暂停
    def _toggle_playback(self) -> None:
        if self.capture is None:
            self._push_status("当前状态：请先加载视频。")
            return

        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("播放视频")
            self._push_status("当前状态：视频已暂停。")
            self._append_log("视频播放已暂停。")
            return

        self.timer.start(self.frame_interval_ms)
        self.play_button.setText("暂停播放")
        self._push_status("当前状态：视频播放中。")
        self._append_log("开始播放当前视频。")

    # 执行检测
    def _run_detection(self) -> None:
        if not self.source_video_path:
            self._push_status("当前状态：请先加载视频。")
            return

        self.detector.refresh_model_path()
        if not self.detector.is_ready():
            self._push_status("当前状态：未找到可用模型权重。")
            self._append_log("检测失败：未找到可用模型权重。")
            return

        self._release_capture()
        self._set_busy(True) # 禁用所有按钮
        self.progress_bar.setValue(0)

        if self.detector.using_fallback_model():
            self._push_status("当前状态：使用基础模型进行流程验证。")
            self._append_log("未找到训练后权重，当前使用基础 yolo11s.pt。")
        else:
            self._push_status("当前状态：正在进行视频人脸检测。")
            self._append_log(f"开始检测，使用权重：{self.detector.model_path.as_posix()}")

        QApplication.processEvents()

        try:  # 实时回传信息
            result = self.detector.detect_video(
                self.source_video_path,
                progress_callback=self._update_detection_progress,
            )
        except Exception as exc:
            self._set_busy(False)
            self._push_status(f"当前状态：检测失败，原因：{exc}")
            self._append_log(f"检测失败：{exc}")
            return

        self.detected_video_path = str(result["output_path"])
        self.preview_video_path = self.detected_video_path   # 预览

        if not self._open_video_for_preview(self.detected_video_path):
            self._set_busy(False)
            self._push_status("当前状态：检测完成，但结果视频预览失败。")
            self._append_log("检测完成，但结果视频预览失败。")
            return

        self.progress_bar.setValue(100)
        self.export_button.setEnabled(True)
        self.play_button.setEnabled(True)
        self.detect_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self._set_busy(False)

        if result["using_fallback_model"]:
            self._push_status("当前状态：检测流程已完成，结果仅用于流程验证。")
            self._append_log(f"检测完成，结果视频：{self.detected_video_path}")
        else:
            self._push_status("当前状态：视频人脸检测完成。")
            self._append_log(f"人脸检测完成，结果视频：{self.detected_video_path}")

        try:  # 存入数据
            save_detection_record(
                username=self.current_user,
                source_video=self.source_video_path,
                result_video=self.detected_video_path,
                model_path=self.detector.model_path.as_posix(),
            )
            self.record_status_label.setText("已写入数据库")
            self._append_log("检测记录已写入数据库。")
        except Exception as exc:
            self.record_status_label.setText("数据库写入失败")
            self._append_log(f"检测记录写入失败：{exc}")

        self._refresh_system_info()
        self._emit_data_changed()

    # 导出
    def _export_result(self) -> None:
        if not self.detected_video_path:
            self._push_status("当前状态：请先完成检测。")
            return

        source_path = Path(self.detected_video_path)
        target_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出检测结果",
            str(source_path),
            "Video Files (*.avi *.mp4 *.mov);;All Files (*)",
        )
        if not target_path:
            return

        shutil.copy2(source_path, target_path)

        self._push_status(f"当前状态：结果已导出到 {target_path}")
        self._append_log(f"已导出检测结果到：{target_path}")

    # 更新进度条
    def _update_detection_progress(self, processed_frames: int, total_frames: int) -> None:
        if total_frames > 0:
            progress = int(processed_frames * 100 / total_frames)
            self.progress_bar.setValue(progress)
            self.current_frame_label.setText(f"{processed_frames} / {total_frames}")
        else:
            self.current_frame_label.setText(str(processed_frames))

        if processed_frames == 1 or processed_frames % 20 == 0 or processed_frames == total_frames:
            if total_frames > 0:
                self._push_status(f"当前状态：正在检测视频，第 {processed_frames}/{total_frames} 帧。")
            else:
                self._push_status(f"当前状态：正在检测视频，已处理 {processed_frames} 帧。")
            QApplication.processEvents()

    # 视频帧获取
    def _play_next_frame(self) -> None:
        if self.capture is None:
            return

        ok, frame = self.capture.read()
        if ok:
            self._show_frame(frame)
            self._update_current_frame_label()
            return

        self.timer.stop()
        self.play_button.setText("重新播放")
        self._display_frame_at(0)
        self._push_status("当前状态：视频播放结束。")
        self._append_log("视频播放结束。")

    # 拖拽条逻辑
    def _handle_slider_pressed(self) -> None:
        if self.capture is None:
            return

        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("播放视频")
    def _handle_slider_moved(self, slider_value: int) -> None:
        if self.capture is None:
            return

        if self._display_frame_at(slider_value - 1):
            self._push_status("当前状态：已更新视频预览位置。")
    def _handle_slider_released(self) -> None:
        if self.capture is None:
            return

        if self._display_frame_at(self.preview_slider.value() - 1):
            self._push_status("当前状态：已更新视频预览位置。")

    # 加载视频
    def _open_video_for_preview(self, video_path: str) -> bool:
        self._release_capture()
        self.capture = cv2.VideoCapture(video_path)
        if not self.capture.isOpened():
            self.capture = None
            return False
        # 计算定时
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_interval_ms = max(1, int(1000 / fps)) if fps and fps > 1 else 40
        self.timer.setInterval(self.frame_interval_ms)

        if not self._display_frame_at(0):
            self._release_capture()
            return False

        self.fps_label.setText(f"{fps:.2f}" if fps and fps > 0 else "--")
        self.frame_count_label.setText(str(total_frames))
        self.preview_slider.setRange(1, max(1, total_frames))
        self.preview_slider.setEnabled(True)
        return True

    # 色彩通道转换
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

    # 刷新信息
    def _refresh_system_info(self) -> None:
        self.detector.refresh_model_path()
        self.record_count_label.setText(str(count_detection_records()))
        self.source_path_label.setText(Path(self.source_video_path).name if self.source_video_path else "未加载")
        self.source_path_label.setToolTip(self.source_video_path if self.source_video_path else "")
        self.result_path_label.setText(Path(self.detected_video_path).name if self.detected_video_path else "未生成")
        self.result_path_label.setToolTip(self.detected_video_path if self.detected_video_path else "")

    # 指定位置
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
        self.current_frame_label.setText(f"{target_frame_index + 1} / {total_frames}")
        self.preview_slider.setValue(target_frame_index + 1)
        return True

    # 计算并显示当前帧
    def _get_current_frame_index(self) -> int:
        if self.capture is None:
            return 0

        current_position = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
        return max(0, current_position - 1)
    def _update_current_frame_label(self) -> None:
        if self.capture is None:
            self.current_frame_label.setText("0 / 0")
            return

        total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = self._get_current_frame_index() + 1 if total_frames > 0 else 0
        self.current_frame_label.setText(f"{current_frame} / {total_frames}")

    # 日志
    def _append_log(self, text: str) -> None:
        self.log_output.appendPlainText(text)

    # 回传数据更改
    def _push_status(self, text: str) -> None:
        self.status_label.setText(text)
        if self.status_callback is not None:
            self.status_callback(text)
    def _emit_data_changed(self) -> None:
        if self.data_changed_callback is not None:
            self.data_changed_callback()

    # 禁用按钮
    def _set_busy(self, busy: bool) -> None:
        self.load_button.setEnabled(not busy)
        self.play_button.setEnabled(not busy and self.capture is not None)
        self.detect_button.setEnabled(not busy and self.detector.is_ready())
        self.export_button.setEnabled(not busy and bool(self.detected_video_path))
        self.preview_slider.setEnabled(not busy and self.capture is not None)

    # 隐藏当前页
    def on_page_hidden(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("播放视频")

    # 对外资源释放
    def release_resources(self) -> None:
        self._release_capture()

    # 统一释放
    def _release_capture(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.preview_slider.setRange(1, 1)
        self.preview_slider.setValue(1)
        self.preview_slider.setEnabled(False)
