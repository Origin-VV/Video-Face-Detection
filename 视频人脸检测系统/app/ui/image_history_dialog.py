import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout


class ImagePreviewDialog(QDialog):
    def __init__(self, image_path: str, title: str, parent=None) -> None:
        super().__init__(parent)
        self.image_path = image_path
        self.setWindowTitle(title)
        self.resize(920, 720)
        self._build_ui()
        self._apply_styles()
        self._open_image()

    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self.path_label = QLabel(self.image_path)
        self.path_label.setWordWrap(True)
        self.path_label.setObjectName("pathLabel")

        self.image_label = QLabel("正在加载图片预览...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(820, 560)
        self.image_label.setObjectName("imagePreview")

        button_row = QHBoxLayout()
        button_row.addStretch()

        self.close_button = QPushButton("关闭")
        self.close_button.setObjectName("secondaryButton")
        self.close_button.clicked.connect(self.accept)
        button_row.addWidget(self.close_button)

        layout.addWidget(self.path_label)
        layout.addWidget(self.image_label, 1)
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
            QLabel#imagePreview {
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
            QPushButton#secondaryButton {
                background-color: #ffffff;
                color: #1f3a56;
                border: 1px solid #c8d5e2;
            }
            """
        )

    def _open_image(self) -> None:
        image = cv2.imread(self.image_path)
        if image is None:
            self.image_label.setText("图片读取失败。")
            return
        self._show_image(image)

    def _show_image(self, image) -> None:
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
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
