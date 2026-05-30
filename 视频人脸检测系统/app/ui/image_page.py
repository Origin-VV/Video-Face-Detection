import shutil
from pathlib import Path
from typing import Callable

import torch
import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QFileDialog,
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

from app.core.database import count_image_detection_records, save_image_detection_record
from app.core.video_detector import VideoDetector


class ImageDetectionPage(QWidget):
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
        self.detector = VideoDetector()
        self.source_image_path = ""
        self.detected_image_path = ""

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

        title = QLabel("图片检测")
        title.setObjectName("moduleTitle")
        title_layout.addWidget(title)
        title_card.setLayout(title_layout)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(14)
        content_layout.addWidget(self._build_left_panel(), 3)
        content_layout.addWidget(self._build_preview_panel(), 7)
        content_layout.addWidget(self._build_right_panel(), 3)

        footer = QFrame()
        footer.setObjectName("footerCard")
        footer_layout = QVBoxLayout()
        footer_layout.setContentsMargins(16, 14, 16, 14)
        footer_layout.setSpacing(8)
        self.status_label = QLabel("当前状态：等待加载图片。")
        self.status_label.setObjectName("statusText")
        footer_layout.addWidget(self.status_label)
        footer.setLayout(footer_layout)

        root_layout.addWidget(title_card)
        root_layout.addLayout(content_layout)
        root_layout.addWidget(footer)
        self.setLayout(root_layout)

    def _build_left_panel(self) -> QWidget:
        panel = QVBoxLayout()
        panel.setSpacing(14)

        control_box = QGroupBox("操作区")
        control_layout = QVBoxLayout()

        self.load_button = QPushButton("加载图片")
        self.detect_button = QPushButton("开始检测")
        self.export_button = QPushButton("导出结果")

        self.load_button.setObjectName("primaryButton")
        self.detect_button.setObjectName("primaryButton")
        self.export_button.setObjectName("secondaryButton")

        self.detect_button.setEnabled(False)
        self.export_button.setEnabled(False)

        self.load_button.clicked.connect(self._choose_image)
        self.detect_button.clicked.connect(self._run_detection)
        self.export_button.clicked.connect(self._export_result)

        control_layout.addWidget(self.load_button)
        control_layout.addWidget(self.detect_button)
        control_layout.addWidget(self.export_button)
        control_box.setLayout(control_layout)

        info_box = QGroupBox("图片信息")
        info_layout = QGridLayout()
        info_layout.setHorizontalSpacing(10)
        info_layout.setVerticalSpacing(8)
        info_layout.setColumnMinimumWidth(0, 76)
        info_layout.setColumnStretch(1, 1)
        self.source_path_label = QLabel("未加载")
        self.result_path_label = QLabel("未生成")
        self.image_size_label = QLabel("--")
        self.face_count_label = QLabel("0")
        self.record_count_label = QLabel("0")
        self.record_status_label = QLabel("尚未写入")

        for label in [self.source_path_label, self.result_path_label]:
            label.setWordWrap(True)

        info_layout.addWidget(QLabel("原始图片"), 0, 0)
        info_layout.addWidget(self.source_path_label, 0, 1)
        info_layout.addWidget(QLabel("结果图片"), 1, 0)
        info_layout.addWidget(self.result_path_label, 1, 1)
        info_layout.addWidget(QLabel("图片尺寸"), 2, 0)
        info_layout.addWidget(self.image_size_label, 2, 1)
        info_layout.addWidget(QLabel("检测人数"), 3, 0)
        info_layout.addWidget(self.face_count_label, 3, 1)
        info_layout.addWidget(QLabel("图片记录"), 4, 0)
        info_layout.addWidget(self.record_count_label, 4, 1)
        info_layout.addWidget(QLabel("写入状态"), 5, 0)
        info_layout.addWidget(self.record_status_label, 5, 1)
        info_box.setLayout(info_layout)

        guide_box = QGroupBox("操作提示")
        guide_layout = QVBoxLayout()
        guide_text = QLabel(
            "1. 加载图片\n"
            "2. 开始检测\n"
            "3. 查看左右预览结果\n"
            "4. 需要时导出图片"
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
    def _build_preview_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("previewCard")
        layout = QHBoxLayout()
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        layout.addWidget(
            self._build_preview_column("原始图片", "加载图片后将在这里预览。", "请先加载图片", "sourcePreview"),
            1,
        )
        layout.addWidget(
            self._build_preview_column("检测结果", "点击开始检测后，将显示带检测框的图片。", "等待检测结果", "resultPreview"),
            1,
        )

        panel.setLayout(layout)
        return panel

    # 预览栏位
    def _build_preview_column(
        self,
        title_text: str,
        hint_text: str,
        placeholder_text: str,
        label_name: str,
    ) -> QWidget:
        column = QFrame()
        column.setObjectName("innerCard")
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        title = QLabel(title_text)
        title.setObjectName("sectionTitle")

        preview = QLabel(placeholder_text)
        preview.setAlignment(Qt.AlignCenter)
        preview.setMinimumSize(420, 520)
        preview.setObjectName(label_name)

        if label_name == "sourcePreview":
            self.source_preview_label = preview
        else:
            self.result_preview_label = preview

        layout.addWidget(title)
        layout.addWidget(preview, 1)
        column.setLayout(layout)
        return column

    # 右侧面板
    def _build_right_panel(self) -> QWidget:
        panel = QVBoxLayout()
        panel.setSpacing(14)

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

    # 刷新信息
    def _refresh_model_info(self) -> None:
        self.detector.refresh_model_path()
        self.record_count_label.setText(str(count_image_detection_records()))

    # 选择图片
    def _choose_image(self) -> None:
        image_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片文件",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp);;All Files (*)",
        )
        if not image_path:
            return

        loaded = self._set_source_image(image_path)
        if not loaded:
            self._append_log(f"图片打开失败：{image_path}")
            self._push_status("当前状态：图片打开失败。")

    # 设置原图
    def _set_source_image(self, image_path: str) -> bool:
        image = cv2.imread(image_path)
        if image is None:
            return False

        self.source_image_path = image_path
        self.detected_image_path = ""
        self.export_button.setEnabled(False)
        self.detect_button.setEnabled(self.detector.is_ready())
        self.source_path_label.setText(Path(image_path).name)
        self.result_path_label.setText("未生成")
        self.face_count_label.setText("0")
        self.record_status_label.setText("尚未写入")
        self.image_size_label.setText(f"{image.shape[1]} x {image.shape[0]}")

        self._show_image(image, self.source_preview_label)
        self.result_preview_label.clear()
        self.result_preview_label.setText("等待检测结果")

        self._refresh_model_info()
        self._append_log(f"已加载图片：{image_path}")
        self._push_status("当前状态：已加载图片，等待开始检测。")
        return True

    # 执行检测
    def _run_detection(self) -> None:
        if not self.source_image_path:
            self._append_log("请先加载图片。")
            self._push_status("当前状态：请先加载图片。")
            return

        self._refresh_model_info()
        if not self.detector.is_ready():
            self._append_log("未找到可用模型权重。")
            self._push_status("当前状态：未找到可用模型权重。")
            return

        if self.detector.using_fallback_model():
            self._append_log("当前使用基础 yolo11s.pt，仅用于流程验证。")
            self._push_status("当前状态：使用基础模型进行流程验证。")
        else:
            self._append_log(f"开始图片检测，使用权重：{self.detector.model_path.as_posix()}")
            self._push_status("当前状态：正在进行图片人脸检测。")

        self._set_busy(True)
        try:
            result = self.detector.detect_image(self.source_image_path)
        except Exception as exc:
            self._set_busy(False)
            self._append_log(f"图片检测失败：{exc}")
            self._push_status(f"当前状态：图片检测失败，原因：{exc}")
            return

        self.detected_image_path = str(result["output_path"])
        result_image = cv2.imread(self.detected_image_path)
        if result_image is None:
            self._set_busy(False)
            self._append_log("检测完成，但结果图片读取失败。")
            self._push_status("当前状态：检测完成，但结果图片读取失败。")
            return

        self._show_image(result_image, self.result_preview_label)
        self.result_path_label.setText(Path(self.detected_image_path).name)
        self.face_count_label.setText(str(result["detection_count"]))
        self.export_button.setEnabled(True)
        self._set_busy(False)

        if result["using_fallback_model"]:
            self._append_log(f"图片检测完成，结果仅用于流程验证：{self.detected_image_path}")
            self._push_status("当前状态：图片检测已完成，结果仅用于流程验证。")
        else:
            self._append_log(
                f"图片检测完成，检测到 {result['detection_count']} 张人脸：{self.detected_image_path}"
            )
            self._push_status("当前状态：图片人脸检测完成。")

        try:
            save_image_detection_record(
                username=self.current_user,
                source_image=self.source_image_path,
                result_image=self.detected_image_path,
                model_path=self.detector.model_path.as_posix(),
            )
            self.record_status_label.setText("已写入数据库")
            self._append_log("图片检测记录已写入数据库。")
        except Exception as exc:
            self.record_status_label.setText("数据库写入失败")
            self._append_log(f"图片检测记录写入失败：{exc}")

        self._refresh_model_info()
        self._emit_data_changed()

    # 导出结果
    def _export_result(self) -> None:
        if not self.detected_image_path:
            self._append_log("请先完成图片检测。")
            self._push_status("当前状态：请先完成图片检测。")
            return

        source_path = Path(self.detected_image_path)
        target_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出检测结果",
            str(source_path),
            "Image Files (*.jpg *.jpeg *.png *.bmp);;All Files (*)",
        )
        if not target_path:
            return

        shutil.copy2(source_path, target_path)
        self._append_log(f"已导出检测结果到：{target_path}")
        self._push_status(f"当前状态：结果已导出到 {target_path}")

    # 显示图片
    def _show_image(self, image, target_label: QLabel) -> None:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_image.shape
        bytes_per_line = channels * width
        qimage = QImage(
            rgb_image.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(qimage)
        scaled = pixmap.scaled(
            target_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        target_label.setPixmap(scaled)

    # 记录日志
    def _append_log(self, text: str) -> None:
        self.log_output.appendPlainText(text)

    # 推送状态
    def _push_status(self, text: str) -> None:
        self.status_label.setText(text)
        if self.status_callback is not None:
            self.status_callback(text)

    # 广播变更
    def _emit_data_changed(self) -> None:
        if self.data_changed_callback is not None:
            self.data_changed_callback()

    # 状态控制
    def _set_busy(self, busy: bool) -> None:
        self.load_button.setEnabled(not busy)
        self.detect_button.setEnabled(not busy and bool(self.source_image_path))
        self.export_button.setEnabled(not busy and bool(self.detected_image_path))