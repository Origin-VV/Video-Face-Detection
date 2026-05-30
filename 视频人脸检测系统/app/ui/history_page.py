import shutil
from pathlib import Path
from typing import Callable

import torch
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)
from app.core.database import (
    count_detection_records,
    count_image_detection_records,
    delete_detection_record,
    delete_image_detection_record,
    fetch_detection_records,
    fetch_image_detection_records,
    get_database_path,
)

from app.ui.history_dialog import HistoryVideoDialog
from app.ui.image_history_dialog import ImagePreviewDialog

# 确认删除对话框
def show_delete_confirm_dialog(parent, title: str, message: str) -> bool:
    dialog = QMessageBox(parent)
    dialog.setIcon(QMessageBox.Question)
    dialog.setWindowTitle(title)
    dialog.setText(message)
    dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    confirm_button = dialog.button(QMessageBox.Yes)
    cancel_button = dialog.button(QMessageBox.No)
    confirm_button.setText("确认")
    cancel_button.setText("取消")
    dialog.setDefaultButton(cancel_button)
    return dialog.exec_() == QMessageBox.Yes


# 视频历史表格
class VideoHistoryTable(QWidget):
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
        self.show_all_records = False
        self.records: list[dict] = []
        self._build_ui()
        self.reload_records()

    # 构建界面
    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(12)

        button_row = QHBoxLayout()
        self.toggle_scope_button = QPushButton("显示全部记录")
        self.preview_button = QPushButton("预览结果视频")
        self.delete_button = QPushButton("删除勾选记录")
        self.refresh_button = QPushButton("刷新记录")

        self.toggle_scope_button.setObjectName("secondaryButton")
        self.preview_button.setObjectName("primaryButton")
        self.delete_button.setObjectName("dangerButton")
        self.refresh_button.setObjectName("secondaryButton")

        self.toggle_scope_button.clicked.connect(self._toggle_scope)
        self.preview_button.clicked.connect(self._preview_selected_video)
        self.delete_button.clicked.connect(self._delete_selected_record)
        self.refresh_button.clicked.connect(self.reload_records)

        button_row.addWidget(self.toggle_scope_button)
        button_row.addWidget(self.preview_button)
        button_row.addWidget(self.delete_button)
        button_row.addWidget(self.refresh_button)
        button_row.addStretch()

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["序号", "用户", "原始视频", "结果视频", "模型权重", "检测时间", "勾选"]
        )
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeToContents)
        self.table.doubleClicked.connect(lambda *_: self._preview_selected_video())

        layout.addLayout(button_row)
        layout.addWidget(self.table, 1)
        self.setLayout(layout)

    # 切换范围
    def _toggle_scope(self) -> None:
        self.show_all_records = not self.show_all_records
        self.toggle_scope_button.setText("只看当前用户" if self.show_all_records else "显示全部记录")
        self.reload_records()

    # 重新加载
    def reload_records(self) -> None:
        username = None if self.show_all_records else self.current_user
        self.records = fetch_detection_records(limit=100, username=username)

        self.table.setRowCount(len(self.records))
        for row_index, record in enumerate(self.records):
            self._set_item(row_index, 0, str(row_index + 1), tooltip=f"数据库ID：{record['id']}")
            self._set_item(row_index, 1, record["username"])
            self._set_path_item(row_index, 2, record["source_video"])
            self._set_path_item(row_index, 3, record["result_video"])
            self._set_path_item(row_index, 4, record["model_path"])
            self._set_item(row_index, 5, record["created_at"])
            self._set_check_item(row_index, 6)

    # 预览视频
    def _preview_selected_video(self) -> None:
        record = self._get_selected_record()
        if record is None:
            QMessageBox.information(self, "未选择记录", "请先在表格中选择一条视频历史记录。")
            return

        video_path = record["result_video"]
        if not Path(video_path).exists():
            QMessageBox.warning(self, "视频不存在", "该记录对应的检测结果视频文件不存在。")
            return

        dialog = HistoryVideoDialog(video_path=video_path, parent=self)
        dialog.exec_()
        self._notify_status("当前状态：已打开视频历史预览。")

    # 删除记录
    def _delete_selected_record(self) -> None:
        checked_records = self._get_checked_records()
        if not checked_records:
            QMessageBox.information(self, "未勾选记录", "请先勾选至少一条视频历史记录。")
            return

        confirmed = show_delete_confirm_dialog(
            self,
            "确认删除",
            f"将删除勾选的 {len(checked_records)} 条视频历史记录，并会删除对应的本地检测结果视频。是否继续？",
        )
        if not confirmed:
            return

        deleted_count = 0
        deleted_video_count = 0
        failed_video_paths: list[str] = []
        for record in checked_records:
            if delete_detection_record(record["id"]):
                deleted_count += 1
                if self._delete_result_video_file(record["result_video"]):
                    deleted_video_count += 1
                elif Path(record["result_video"]).exists():
                    failed_video_paths.append(record["result_video"])

        if deleted_count == 0:
            QMessageBox.warning(self, "删除失败", "未能删除勾选的视频历史记录。")
            return

        self.reload_records()
        self._emit_data_changed()
        if deleted_count < len(checked_records):
            QMessageBox.warning(
                self,
                "部分删除成功",
                f"共勾选 {len(checked_records)} 条视频历史记录，成功删除 {deleted_count} 条；成功删除 {deleted_video_count} 个本地结果视频。",
            )
        elif failed_video_paths:
            QMessageBox.warning(
                self,
                "视频文件删除不完整",
                f"数据库记录已删除 {deleted_count} 条，但有 {len(failed_video_paths)} 个本地结果视频删除失败。",
            )
        self._notify_status(
            f"当前状态：已删除 {deleted_count} 条视频历史记录，并删除 {deleted_video_count} 个本地结果视频。"
        )

    # 获取选中项
    def _get_selected_record(self) -> dict | None:
        current_row = self.table.currentRow()
        if current_row < 0 or current_row >= len(self.records):
            return None
        return self.records[current_row]

    # 获取勾选项
    def _get_checked_records(self) -> list[dict]:
        checked_records: list[dict] = []
        for row_index, record in enumerate(self.records):
            item = self.table.item(row_index, 6)
            if item is not None and item.checkState() == Qt.Checked:
                checked_records.append(record)
        return checked_records

    # 删除视频文件
    def _delete_result_video_file(self, video_path: str) -> bool:
        path = Path(video_path)
        if not video_path or not path.exists():
            return True

        try:
            path.unlink()
            return True
        except OSError:
            return False

    # 推送状态
    def _notify_status(self, text: str) -> None:
        if self.status_callback is not None:
            self.status_callback(text)

    # 广播变更
    def _emit_data_changed(self) -> None:
        if self.data_changed_callback is not None:
            self.data_changed_callback()

    # 设置单元格
    def _set_item(self, row: int, column: int, text: str, tooltip: str | None = None) -> None:
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)
        if tooltip:
            item.setToolTip(tooltip)
        self.table.setItem(row, column, item)

    # 设置路径单元格
    def _set_path_item(self, row: int, column: int, path_text: str) -> None:
        display_text = Path(path_text).name if path_text else ""
        item = QTableWidgetItem(display_text)
        item.setToolTip(path_text)
        self.table.setItem(row, column, item)

    # 设置勾选框
    def _set_check_item(self, row: int, column: int) -> None:
        item = QTableWidgetItem()
        item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        item.setCheckState(Qt.Unchecked)
        item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, column, item)


# 图片历史表格
class ImageHistoryTable(QWidget):
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
        self.show_all_records = False
        self.records: list[dict] = []
        self._build_ui()
        self.reload_records()

    # 构建界面
    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(12)

        button_row = QHBoxLayout()
        self.toggle_scope_button = QPushButton("显示全部记录")
        self.preview_source_button = QPushButton("预览原始图片")
        self.preview_result_button = QPushButton("预览结果图片")
        self.delete_button = QPushButton("删除勾选记录")
        self.refresh_button = QPushButton("刷新记录")

        self.toggle_scope_button.setObjectName("secondaryButton")
        self.preview_source_button.setObjectName("secondaryButton")
        self.preview_result_button.setObjectName("primaryButton")
        self.delete_button.setObjectName("dangerButton")
        self.refresh_button.setObjectName("secondaryButton")

        self.toggle_scope_button.clicked.connect(self._toggle_scope)
        self.preview_source_button.clicked.connect(self._preview_source_image)
        self.preview_result_button.clicked.connect(self._preview_result_image)
        self.delete_button.clicked.connect(self._delete_selected_record)
        self.refresh_button.clicked.connect(self.reload_records)

        button_row.addWidget(self.toggle_scope_button)
        button_row.addWidget(self.preview_source_button)
        button_row.addWidget(self.preview_result_button)
        button_row.addWidget(self.delete_button)
        button_row.addWidget(self.refresh_button)
        button_row.addStretch()

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["序号", "用户", "原始图片", "结果图片", "模型权重", "检测时间", "勾选"]
        )
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeToContents)
        self.table.doubleClicked.connect(lambda *_: self._preview_result_image())

        layout.addLayout(button_row)
        layout.addWidget(self.table, 1)
        self.setLayout(layout)

    # 切换范围
    def _toggle_scope(self) -> None:
        self.show_all_records = not self.show_all_records
        self.toggle_scope_button.setText("只看当前用户" if self.show_all_records else "显示全部记录")
        self.reload_records()

    # 重新加载
    def reload_records(self) -> None:
        username = None if self.show_all_records else self.current_user
        self.records = fetch_image_detection_records(limit=100, username=username)

        self.table.setRowCount(len(self.records))
        for row_index, record in enumerate(self.records):
            self._set_item(row_index, 0, str(row_index + 1), tooltip=f"数据库ID：{record['id']}")
            self._set_item(row_index, 1, record["username"])
            self._set_path_item(row_index, 2, record["source_image"])
            self._set_path_item(row_index, 3, record["result_image"])
            self._set_path_item(row_index, 4, record["model_path"])
            self._set_item(row_index, 5, record["created_at"])
            self._set_check_item(row_index, 6)

    # 预览原图
    def _preview_source_image(self) -> None:
        self._preview_selected_image(record_key="source_image", title="原始图片预览")

    # 预览结果图
    def _preview_result_image(self) -> None:
        self._preview_selected_image(record_key="result_image", title="结果图片预览")

    # 指定图片预览
    def _preview_selected_image(self, record_key: str, title: str) -> None:
        record = self._get_selected_record()
        if record is None:
            QMessageBox.information(self, "未选择记录", "请先在表格中选择一条图片历史记录。")
            return

        image_path = record[record_key]
        if not Path(image_path).exists():
            QMessageBox.warning(self, "图片不存在", "该记录对应的图片文件不存在。")
            return

        dialog = ImagePreviewDialog(image_path=image_path, title=title, parent=self)
        dialog.exec_()
        self._notify_status("当前状态：已打开图片历史预览。")

    # 删除记录
    def _delete_selected_record(self) -> None:
        checked_records = self._get_checked_records()
        if not checked_records:
            QMessageBox.information(self, "未勾选记录", "请先勾选至少一条图片历史记录。")
            return

        confirmed = show_delete_confirm_dialog(
            self,
            "确认删除",
            f"将删除勾选的 {len(checked_records)} 条图片历史记录在数据库中的信息，但不会删除磁盘上的图片文件。是否继续？",
        )
        if not confirmed:
            return

        deleted_count = 0
        for record in checked_records:
            if delete_image_detection_record(record["id"]):
                deleted_count += 1

        if deleted_count == 0:
            QMessageBox.warning(self, "删除失败", "未能删除勾选的图片历史记录。")
            return

        self.reload_records()
        self._emit_data_changed()
        if deleted_count < len(checked_records):
            QMessageBox.warning(
                self,
                "部分删除成功",
                f"共勾选 {len(checked_records)} 条图片历史记录，成功删除 {deleted_count} 条。",
            )
        self._notify_status(f"当前状态：已删除 {deleted_count} 条图片历史记录。")

    # 获取选中项
    def _get_selected_record(self) -> dict | None:
        current_row = self.table.currentRow()
        if current_row < 0 or current_row >= len(self.records):
            return None
        return self.records[current_row]

    # 获取勾选项
    def _get_checked_records(self) -> list[dict]:
        checked_records: list[dict] = []
        for row_index, record in enumerate(self.records):
            item = self.table.item(row_index, 6)
            if item is not None and item.checkState() == Qt.Checked:
                checked_records.append(record)
        return checked_records

    # 推送状态
    def _notify_status(self, text: str) -> None:
        if self.status_callback is not None:
            self.status_callback(text)

    # 广播变更
    def _emit_data_changed(self) -> None:
        if self.data_changed_callback is not None:
            self.data_changed_callback()

    # 设置单元格
    def _set_item(self, row: int, column: int, text: str, tooltip: str | None = None) -> None:
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)
        if tooltip:
            item.setToolTip(tooltip)
        self.table.setItem(row, column, item)

    # 设置路径单元格
    def _set_path_item(self, row: int, column: int, path_text: str) -> None:
        display_text = Path(path_text).name if path_text else ""
        item = QTableWidgetItem(display_text)
        item.setToolTip(path_text)
        self.table.setItem(row, column, item)

    # 设置勾选框
    def _set_check_item(self, row: int, column: int) -> None:
        item = QTableWidgetItem()
        item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        item.setCheckState(Qt.Unchecked)
        item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, column, item)


# 历史中心页面
class HistoryCenterPage(QWidget):
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
        self._build_ui()
        self.reload_all()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(14)

        title_card = QFrame()
        title_card.setObjectName("moduleHeroCard")
        title_layout = QVBoxLayout()
        title_layout.setContentsMargins(18, 18, 18, 18)
        title_layout.setSpacing(6)

        title = QLabel("历史记录中心")
        title.setObjectName("moduleTitle")
        title_layout.addWidget(title)
        title_card.setLayout(title_layout)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(14)

        main_card = QFrame()
        main_card.setObjectName("previewCard")
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(12)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("historyTabs")
        self.video_table = VideoHistoryTable(
            current_user=self.current_user,
            status_callback=self._notify_status,
            data_changed_callback=self._emit_data_changed,
        )
        self.image_table = ImageHistoryTable(
            current_user=self.current_user,
            status_callback=self._notify_status,
            data_changed_callback=self._emit_data_changed,
        )
        self.tabs.addTab(self.video_table, "视频记录")
        self.tabs.addTab(self.image_table, "图片记录")

        main_layout.addWidget(self.tabs)
        main_card.setLayout(main_layout)

        side_panel = QVBoxLayout()
        side_panel.setSpacing(14)

        summary_box = QGroupBox("记录概览")
        summary_layout = QGridLayout()
        self.video_count_label = QLabel("0")
        self.image_count_label = QLabel("0")
        self.database_label = QLabel(get_database_path().as_posix())
        self.database_label.setWordWrap(True)
        summary_layout.addWidget(QLabel("视频记录"), 0, 0)
        summary_layout.addWidget(self.video_count_label, 0, 1)
        summary_layout.addWidget(QLabel("图片记录"), 1, 0)
        summary_layout.addWidget(self.image_count_label, 1, 1)
        summary_layout.addWidget(QLabel("数据库"), 2, 0)
        summary_layout.addWidget(self.database_label, 2, 1)
        summary_box.setLayout(summary_layout)

        guide_box = QGroupBox("操作提示")
        guide_layout = QVBoxLayout()
        guide_text = QLabel(
            "1. 在两个标签页之间切换\n"
            "2. 支持查看当前用户或全部记录\n"
            "3. 可预览结果并删除记录\n"
            "4. 删除不会影响原始文件"
        )
        guide_text.setWordWrap(True)
        guide_layout.addWidget(guide_text)
        guide_box.setLayout(guide_layout)

        side_panel.addWidget(summary_box)
        side_panel.addWidget(guide_box)
        side_panel.addStretch()

        content_layout.addWidget(main_card, 7)
        side_container = QWidget()
        side_container.setLayout(side_panel)
        content_layout.addWidget(side_container, 3)

        root_layout.addWidget(title_card)
        root_layout.addLayout(content_layout)
        self.setLayout(root_layout)

    # 重新加载全部
    def reload_all(self) -> None:
        self.video_table.reload_records()
        self.image_table.reload_records()
        self.video_count_label.setText(str(count_detection_records()))
        self.image_count_label.setText(str(count_image_detection_records()))

    # 设置当前标签
    def set_current_tab(self, index: int) -> None:
        self.tabs.setCurrentIndex(index)

    # 推送状态
    def _notify_status(self, text: str) -> None:
        if self.status_callback is not None:
            self.status_callback(text)

    # 广播变更
    def _emit_data_changed(self) -> None:
        self.reload_all()
        if self.data_changed_callback is not None:
            self.data_changed_callback()