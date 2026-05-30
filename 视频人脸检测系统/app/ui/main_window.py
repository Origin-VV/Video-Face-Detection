from pathlib import Path

import torch
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.core.database import count_detection_records, count_image_detection_records, get_database_path
from app.core.video_detector import VideoDetector
from app.ui.workbench_pages import CameraDetectionPage, HistoryCenterPage, ImageDetectionPage, VideoDetectionPage


class MainWindow(QWidget):
    def __init__(self, username: str = "admin", relogin_callback=None) -> None:
        super().__init__()
        self.current_user = username
        self.relogin_callback = relogin_callback
        self.detector = VideoDetector()
        self.current_page_index = 0

        self.setWindowTitle("视频人脸检测系统")
        self.resize(1520, 940)
        self._build_ui()
        self._apply_styles()
        self._refresh_dashboard()
        self._switch_page(0)

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(14)

        root_layout.addWidget(self._build_header())
        root_layout.addWidget(self._build_summary_cards())

        workspace_layout = QHBoxLayout()
        workspace_layout.setSpacing(14)
        workspace_layout.addWidget(self._build_navigation(), 2)
        workspace_layout.addWidget(self._build_workspace(), 10)

        root_layout.addLayout(workspace_layout)
        root_layout.addWidget(self._build_footer())
        self.setLayout(root_layout)

    # 顶部
    def _build_header(self) -> QWidget:

        header = QFrame()
        header.setObjectName("headerCard")
        layout = QHBoxLayout()
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(18)

        title_block = QVBoxLayout()
        title = QLabel("视频人脸检测系统")
        title.setObjectName("pageTitle")
        title.setAlignment(Qt.AlignCenter)
        title_block.addWidget(title)
        title_block.setContentsMargins(0, 0, 0, 0)

        user_card = QFrame()
        user_card.setObjectName("userCard")
        user_card.setMinimumWidth(260)
        user_card.setMaximumWidth(320)
        user_block = QVBoxLayout()
        user_block.setContentsMargins(16, 14, 16, 14)
        user_block.setSpacing(8)

        user_title = QLabel("当前用户")
        user_title.setObjectName("userCardTitle")
        self.user_label = QLabel(self.current_user)
        self.user_label.setObjectName("userCardValue")

        button_row = QHBoxLayout()
        button_row.addStretch()
        self.logout_button = QPushButton("退出登录")
        self.logout_button.setObjectName("secondaryButton")
        self.logout_button.clicked.connect(self._logout)
        button_row.addWidget(self.logout_button)

        user_block.addWidget(user_title)
        user_block.addWidget(self.user_label)
        user_block.addLayout(button_row)
        user_card.setLayout(user_block)

        left_placeholder = QWidget()
        left_placeholder.setMinimumWidth(260)
        left_placeholder.setMaximumWidth(320)

        title_container = QWidget()
        title_container.setObjectName("titleContainer")
        title_container.setLayout(title_block)

        layout.addWidget(left_placeholder, 0, Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(title_container, 1, Qt.AlignCenter)
        layout.addWidget(user_card, 0, Qt.AlignRight | Qt.AlignVCenter)
        header.setLayout(layout)
        return header

    # 信息板
    def _build_summary_cards(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        self.model_card = self._create_stat_card("当前模型", "等待检测", "正在使用的检测权重")
        self.video_card = self._create_stat_card("视频记录", "0", "视频历史总数")
        self.image_card = self._create_stat_card("图片记录", "0", "图片历史总数")
        self.db_card = self._create_stat_card("数据库", "-", "本地 SQLite 文件")

        layout.addWidget(self.model_card["frame"], 3)
        layout.addWidget(self.video_card["frame"], 2)
        layout.addWidget(self.image_card["frame"], 2)
        layout.addWidget(self.db_card["frame"], 3)
        container.setLayout(layout)
        return container

    def _create_stat_card(self, title: str, value: str, _note: str) -> dict:
        frame = QFrame()
        frame.setObjectName("statCard")
        layout = QVBoxLayout()
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(4)

        title_label = QLabel(title)
        title_label.setObjectName("statTitle")
        value_label = QLabel(value)
        value_label.setObjectName("statValue")
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        frame.setLayout(layout)
        return {"frame": frame, "value": value_label}

    # 左侧导航栏
    def _build_navigation(self) -> QWidget:
        nav = QFrame()
        nav.setObjectName("navCard")
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        nav_title = QLabel("导航")
        nav_title.setObjectName("navTitle")

        self.video_nav_button = QPushButton("视频检测")
        self.image_nav_button = QPushButton("图片检测")
        self.camera_nav_button = QPushButton("摄像头检测")
        self.history_nav_button = QPushButton("历史记录")
        self.nav_buttons = [
            self.video_nav_button,
            self.image_nav_button,
            self.camera_nav_button,
            self.history_nav_button,
        ]

        self.video_nav_button.clicked.connect(lambda: self._switch_page(0))
        self.image_nav_button.clicked.connect(lambda: self._switch_page(1))
        self.camera_nav_button.clicked.connect(lambda: self._switch_page(2))
        self.history_nav_button.clicked.connect(lambda: self._switch_page(3))

        for button in self.nav_buttons:
            button.setObjectName("navButton")

        layout.addWidget(nav_title)
        layout.addWidget(self.video_nav_button)
        layout.addWidget(self.image_nav_button)
        layout.addWidget(self.camera_nav_button)
        layout.addWidget(self.history_nav_button)
        layout.addStretch()
        nav.setLayout(layout)
        return nav

    # 工作区
    def _build_workspace(self) -> QWidget:
        self.workspace_frame = QFrame()
        self.workspace_frame.setObjectName("workspaceCard")
        layout = QVBoxLayout()
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)
        # 工作栈
        self.page_stack = QStackedWidget()
        self.video_page = VideoDetectionPage(
            current_user=self.current_user,
            status_callback=self._set_global_status,
            data_changed_callback=self._handle_data_changed,
        )
        self.image_page = ImageDetectionPage(
            current_user=self.current_user,
            status_callback=self._set_global_status,
            data_changed_callback=self._handle_data_changed,
        )
        self.camera_page = CameraDetectionPage(
            current_user=self.current_user,
            status_callback=self._set_global_status,
            data_changed_callback=self._handle_data_changed,
        )
        self.history_page = HistoryCenterPage(
            current_user=self.current_user,
            status_callback=self._set_global_status,
            data_changed_callback=self._handle_data_changed,
        )

        self.page_stack.addWidget(self.video_page)
        self.page_stack.addWidget(self.image_page)
        self.page_stack.addWidget(self.camera_page)
        self.page_stack.addWidget(self.history_page)

        layout.addWidget(self.page_stack)
        self.workspace_frame.setLayout(layout)
        return self.workspace_frame

    # 底部状态栏
    def _build_footer(self) -> QWidget:
        footer = QFrame()
        footer.setObjectName("footerCard")
        layout = QHBoxLayout()
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(12)

        self.global_status_label = QLabel("当前状态：系统已就绪。")
        self.global_status_label.setObjectName("statusText")
        layout.addWidget(self.global_status_label, 1)

        self.footer_path_label = QLabel(get_database_path().as_posix())
        self.footer_path_label.setObjectName("footerPath")
        layout.addWidget(self.footer_path_label)

        footer.setLayout(layout)
        return footer

    # 页面切换
    def _switch_page(self, page_index: int, history_tab_index: int | None = None) -> None:
        if self.current_page_index == 0 and page_index != 0:
            self.video_page.on_page_hidden()
        if self.current_page_index == 2 and page_index != 2:
            self.camera_page.on_page_hidden()

        self.current_page_index = page_index
        self.page_stack.setCurrentIndex(page_index)

        if page_index == 3 and history_tab_index is not None:
            self.history_page.set_current_tab(history_tab_index)

        self._set_active_nav_button(page_index)

        if page_index == 2:
            self._set_global_status("当前状态：已进入摄像头检测。")
            self.history_page.reload_all()
            return
        if page_index == 3:
            self._set_global_status("当前状态：已进入历史记录。")
            self.history_page.reload_all()
            return

        page_titles = {
            0: "当前状态：已进入视频检测。",
            1: "当前状态：已进入图片检测。",
            2: "当前状态：已进入摄像头检测。",
            3: "当前状态：已进入历史记录。",
        }
        self._set_global_status(page_titles.get(page_index, "当前状态：工作台已切换页面。"))
        self.history_page.reload_all()

    # 高亮选中的导航栏
    def _set_active_nav_button(self, page_index: int) -> None:
        for index, button in enumerate(self.nav_buttons):
            button.setObjectName("navButtonActive" if index == page_index else "navButton")
            button.style().unpolish(button)
            button.style().polish(button)
            button.update()

    # 刷新
    def _refresh_dashboard(self) -> None:
        self.detector.refresh_model_path()

        if self.detector.is_ready():
            self.model_card["value"].setText(self.detector.model_path.name)
        else:
            self.model_card["value"].setText("未找到模型")

        self.video_card["value"].setText(str(count_detection_records()))
        self.image_card["value"].setText(str(count_image_detection_records()))
        self.db_card["value"].setText(Path(get_database_path()).name)
        self.footer_path_label.setText(get_database_path().as_posix())

    # 信息刷新
    def _handle_data_changed(self) -> None:
        self._refresh_dashboard()
        self.history_page.reload_all()
        self.video_page._refresh_system_info()
        self.image_page._refresh_model_info()
        self.camera_page._refresh_model_info()

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background-color: #e8f0f5;
                color: #223247;
                font-size: 16px;
            }
            QLabel {
                background-color: transparent;
            }
            QFrame#headerCard, QFrame#statCard, QFrame#navCard, QFrame#workspaceCard, QFrame#footerCard,
            QFrame#moduleHeroCard, QFrame#previewCard, QFrame#innerCard, QFrame#navInfoCard, QGroupBox {
                background-color: white;
                border: 1px solid #d5e0ea;
                border-radius: 16px;
            }
            QFrame#userCard {
                background-color: #f7fafc;
                border: 1px solid #d5e0ea;
                border-radius: 14px;
            }
            QFrame#moduleHeroCard {
                background-color: #f7fafc;
            }
            QGroupBox {
                margin-top: 12px;
                padding: 14px;
                font-weight: 700;
                font-size: 17px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }
            QWidget#titleContainer {
                background-color: transparent;
            }
            QLabel#pageTitle {
                font-size: 35px;
                font-weight: 700;
                color: #17324d;
            }
            QLabel#pageSubtitle, QLabel#sectionHint, QLabel#moduleSubtitle, QLabel#statNote {
                font-size: 17px;
                color: #607284;
            }
            QLabel#moduleTitle {
                font-size: 26px;
                font-weight: 700;
                color: #17324d;
            }
            QLabel#sectionTitle, QLabel#navTitle {
                font-size: 21px;
                font-weight: 700;
                color: #17324d;
            }
            QLabel#userCardTitle {
                font-size: 15px;
                font-weight: 700;
                color: #607284;
            }
            QLabel#userCardValue {
                font-size: 22px;
                font-weight: 700;
                color: #17324d;
            }
            QLabel#infoBadge {
                background-color: #f5f9fc;
                border: 1px solid #d8e2ec;
                border-radius: 10px;
                padding: 8px 12px;
            }
            QLabel#alarmStatusNormal {
                color: #17324d;
                font-size: 18px;
                font-weight: 700;
            }
            QLabel#alarmStatusAlert {
                background-color: #d92d20;
                color: white;
                border: 2px solid #a40000;
                border-radius: 10px;
                padding: 10px 14px;
                font-size: 20px;
                font-weight: 700;
            }
            QLabel#statTitle {
                font-size: 16px;
                font-weight: 700;
                color: #5f7387;
            }
            QLabel#statValue {
                font-size: 24px;
                font-weight: 700;
                color: #15314b;
            }
            QLabel#statusText {
                font-size: 17px;
                font-weight: 600;
                color: #17324d;
            }
            QLabel#footerPath {
                font-size: 15px;
                color: #5f7387;
            }
            QLabel#videoPreview, QLabel#sourcePreview, QLabel#resultPreview {
                border: 2px dashed #b8c8d8;
                border-radius: 18px;
                background-color: #101820;
                color: white;
            }
            QPushButton {
                min-height: 40px;
                border-radius: 12px;
                font-size: 16px;
                font-weight: 600;
                padding: 0 14px;
            }
            QPushButton#primaryButton {
                background-color: #1677ff;
                color: white;
                border: none;
            }
            QPushButton#secondaryButton {
                background-color: #ffffff;
                color: #1f3a56;
                border: 1px solid #cad7e3;
            }
            QPushButton#dangerButton {
                background-color: #fff5f5;
                color: #c0392b;
                border: 1px solid #f0c5bf;
            }
            QPushButton:disabled {
                background-color: #f3f4f6;
                color: #9ca3af;
                border: 1px solid #e5e7eb;
            }
            QPushButton#navButton {
                background-color: #f7fafc;
                color: #17324d;
                border: 1px solid #d4e0ea;
                text-align: left;
                padding-left: 16px;
            }
            QPushButton#navButtonActive {
                background-color: #17324d;
                color: white;
                border: 1px solid #17324d;
                text-align: left;
                padding-left: 16px;
            }
            QPlainTextEdit {
                background-color: #fbfdff;
                border: 1px solid #d8e2ec;
                border-radius: 12px;
                font-size: 16px;
                padding: 8px;
            }
            QProgressBar {
                border: 1px solid #d8e2ec;
                border-radius: 8px;
                background-color: #f4f7fb;
                text-align: center;
                min-height: 22px;
            }
            QProgressBar::chunk {
                background-color: #1677ff;
                border-radius: 7px;
            }
            QTableWidget {
                background-color: white;
                border: 1px solid #d8e2ec;
                border-radius: 12px;
                font-size: 16px;
                gridline-color: #e7edf3;
            }
            QTableWidget::item {
                padding: 6px;
            }
            QHeaderView::section {
                background-color: #eef4f9;
                color: #17324d;
                font-weight: 700;
                border: none;
                border-bottom: 1px solid #d8e2ec;
                padding: 8px;
            }
            QTabWidget::pane {
                border: none;
            }
            QTabBar::tab {
                background-color: #f3f7fb;
                border: 1px solid #d6e1eb;
                border-radius: 10px;
                padding: 10px 18px;
                margin-right: 6px;
            }
            QTabBar::tab:selected {
                background-color: #17324d;
                color: white;
                border-color: #17324d;
            }
            """
        )

    def _set_global_status(self, text: str) -> None:
        self.global_status_label.setText(text)

    def _logout(self) -> None:
        self._set_global_status("当前状态：用户已退出登录。")
        self.close()
        if self.relogin_callback is not None:
            self.relogin_callback()

    # 安全关闭
    def closeEvent(self, event) -> None:
        self.video_page.release_resources()
        self.camera_page.release_resources()
        super().closeEvent(event)
