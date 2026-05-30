from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core.database import (
    DEFAULT_PASSWORD,
    DEFAULT_USERNAME,
    authenticate_user,
    create_user,
)

# 登录窗口
class LoginWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.main_window = None
        self.register_dialog = None
        self.setWindowTitle("视频人脸检测系统登录")
        self.resize(960, 640)
        self._build_ui()
        self._apply_styles()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(40, 40, 40, 40)

        header = QLabel("基于深度学习的视频人脸检测系统")
        header.setObjectName("loginHeader")
        header.setAlignment(Qt.AlignCenter)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(24)

        intro_card = QFrame()
        intro_card.setObjectName("sideCard")
        intro_layout = QVBoxLayout()

        intro_title = QLabel("系统功能")
        intro_title.setObjectName("cardTitle")

        intro_text = QLabel(
            "1. 本地视频加载与预览\n"
            "2. 使用训练后的人脸模型进行检测\n"
            "3. 导出检测结果视频\n"
            "4. 查看模型与处理状态\n"
            "5. 用户信息和检测记录保存在本地数据库"
        )
        intro_text.setWordWrap(True)
        intro_text.setObjectName("cardBody")

        demo_title = QLabel("默认账号")
        demo_title.setObjectName("cardTitle")
        demo_text = QLabel(
            f"用户名：{DEFAULT_USERNAME}\n"
            f"密码：{DEFAULT_PASSWORD}"
        )
        demo_text.setObjectName("cardBody")

        intro_layout.addWidget(intro_title)
        intro_layout.addWidget(intro_text)
        intro_layout.addSpacing(16)
        intro_layout.addWidget(demo_title)
        intro_layout.addWidget(demo_text)
        intro_layout.addStretch()
        intro_card.setLayout(intro_layout)

        form_card = QFrame()
        form_card.setObjectName("formCard")
        form_layout = QVBoxLayout()
        form_layout.setSpacing(14)

        form_title = QLabel("用户登录")
        form_title.setObjectName("formTitle")
        form_title.setAlignment(Qt.AlignCenter)

        username_label = QLabel("用户名")
        username_label.setObjectName("formLabel")
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("请输入用户名")

        password_label = QLabel("密码")
        password_label.setObjectName("formLabel")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("请输入密码")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.returnPressed.connect(self._handle_login)

        self.message_label = QLabel("请输入账号和密码后登录。")
        self.message_label.setObjectName("messageLabel")
        self.message_label.setAlignment(Qt.AlignCenter)

        button_row = QHBoxLayout()
        self.login_button = QPushButton("登录系统")
        self.login_button.setObjectName("primaryButton")
        self.register_button = QPushButton("注册用户")
        self.register_button.setObjectName("secondaryButton")
        self.clear_button = QPushButton("清空输入")
        self.clear_button.setObjectName("secondaryButton")

        self.login_button.clicked.connect(self._handle_login)
        self.register_button.clicked.connect(self._open_register_dialog)
        self.clear_button.clicked.connect(self._clear_inputs)

        button_row.addWidget(self.login_button)
        button_row.addWidget(self.register_button)
        button_row.addWidget(self.clear_button)

        form_layout.addWidget(form_title)
        form_layout.addWidget(username_label)
        form_layout.addWidget(self.username_input)
        form_layout.addWidget(password_label)
        form_layout.addWidget(self.password_input)
        form_layout.addWidget(self.message_label)
        form_layout.addLayout(button_row)
        form_layout.addStretch()
        form_card.setLayout(form_layout)

        content_layout.addWidget(intro_card, 3)
        content_layout.addWidget(form_card, 2)

        root_layout.addWidget(header)
        root_layout.addSpacing(24)
        root_layout.addLayout(content_layout)
        self.setLayout(root_layout)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background-color: #eef3f8;
                color: #233245;
                font-size: 18px;
            }
            QLabel {
                background-color: transparent;
            }
            QFrame#sideCard, QFrame#formCard {
                background-color: white;
                border: 1px solid #d9e2ec;
                border-radius: 18px;
                padding: 18px;
            }
            QLabel#loginHeader {
                font-size: 32px;
                font-weight: 700;
                color: #17324d;
            }
            QLabel#loginSubtitle {
                font-size: 19px;
                color: #5a6b7f;
            }
            QLabel#cardTitle, QLabel#formTitle {
                font-size: 24px;
                font-weight: 700;
                color: #17324d;
            }
            QLabel#formLabel {
                font-size: 18px;
                font-weight: 700;
                color: #17324d;
            }
            QLabel#cardBody, QLabel#messageLabel {
                font-size: 18px;
                color: #526477;
                line-height: 1.5;
            }
            QLineEdit {
                min-height: 38px;
                border: 1px solid #c8d5e2;
                border-radius: 10px;
                font-size: 18px;
                padding: 0 12px;
                background-color: #f8fbfd;
            }
            QPushButton {
                min-height: 40px;
                border-radius: 10px;
                font-size: 18px;
                font-weight: 600;
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

    # 登录主界面
    def _handle_login(self) -> None:
        username = self.username_input.text().strip()
        password = self.password_input.text()

        if authenticate_user(username, password):
            from app.ui.main_window import MainWindow

            self.message_label.setText("登录成功，正在进入系统工作台。")
            self.main_window = MainWindow(username=username, relogin_callback=self._show_again)
            self.main_window.show()
            self.hide()
            return

        self.message_label.setText("用户名或密码错误，请重新输入。")
        QMessageBox.warning(self, "登录失败", "用户名或密码错误。")
        self.password_input.clear()
        self.password_input.setFocus()

    def _open_register_dialog(self) -> None:
        self.register_dialog = RegisterDialog(self)
        if self.register_dialog.exec_() == QDialog.Accepted:
            self.username_input.setText(self.register_dialog.registered_username)
            self.password_input.clear()
            self.message_label.setText("注册成功，请使用新账号登录。")
            self.password_input.setFocus()

    def _clear_inputs(self) -> None:
        self.username_input.clear()
        self.password_input.clear()
        self.message_label.setText("请输入账号和密码后登录。")
        self.username_input.setFocus()

    def _show_again(self) -> None:
        self.password_input.clear()
        self.message_label.setText("请输入账号和密码后登录。")
        self.show()

# 注册窗口
class RegisterDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("注册用户")
        self.resize(420, 300)
        self.registered_username = ""
        self._build_ui()
        self._apply_styles()

    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        title = QLabel("创建新账号")
        title.setObjectName("dialogTitle")
        title.setAlignment(Qt.AlignCenter)

        hint = QLabel("用户名至少 3 位，密码至少 6 位。")
        hint.setObjectName("dialogHint")
        hint.setAlignment(Qt.AlignCenter)

        username_label = QLabel("用户名")
        username_label.setObjectName("formLabel")
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("请输入新用户名")

        password_label = QLabel("密码")
        password_label.setObjectName("formLabel")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("请输入密码")
        self.password_input.setEchoMode(QLineEdit.Password)

        confirm_label = QLabel("确认密码")
        confirm_label.setObjectName("formLabel")
        self.confirm_input = QLineEdit()
        self.confirm_input.setPlaceholderText("请再次输入密码")
        self.confirm_input.setEchoMode(QLineEdit.Password)
        self.confirm_input.returnPressed.connect(self._handle_register)

        self.message_label = QLabel("注册后可直接返回登录。")
        self.message_label.setObjectName("messageLabel")
        self.message_label.setAlignment(Qt.AlignCenter)

        button_row = QHBoxLayout()
        self.register_button = QPushButton("确认注册")
        self.register_button.setObjectName("primaryButton")
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setObjectName("secondaryButton")

        self.register_button.clicked.connect(self._handle_register)
        self.cancel_button.clicked.connect(self.reject)

        button_row.addWidget(self.register_button)
        button_row.addWidget(self.cancel_button)

        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addWidget(username_label)
        layout.addWidget(self.username_input)
        layout.addWidget(password_label)
        layout.addWidget(self.password_input)
        layout.addWidget(confirm_label)
        layout.addWidget(self.confirm_input)
        layout.addWidget(self.message_label)
        layout.addLayout(button_row)
        self.setLayout(layout)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background-color: #f4f8fb;
                color: #223247;
                font-size: 18px;
            }
            QLabel {
                background-color: transparent;
            }
            QLabel#dialogTitle {
                font-size: 26px;
                font-weight: 700;
                color: #17324d;
            }
            QLabel#dialogHint, QLabel#messageLabel {
                font-size: 18px;
                color: #607284;
            }
            QLabel#formLabel {
                font-size: 18px;
                font-weight: 700;
                color: #17324d;
            }
            QLineEdit {
                min-height: 38px;
                border: 1px solid #c8d5e2;
                border-radius: 10px;
                font-size: 18px;
                padding: 0 12px;
                background-color: white;
            }
            QPushButton {
                min-height: 40px;
                border-radius: 10px;
                font-size: 18px;
                font-weight: 600;
            }
            QPushButton#primaryButton {
                background-color: #1f6feb;
                color: white;
                border: none;
            }
            QPushButton#secondaryButton {
                background-color: white;
                color: #1f3a56;
                border: 1px solid #c8d5e2;
            }
            """
        )

    def _handle_register(self) -> None:
        username = self.username_input.text().strip()
        password = self.password_input.text()
        confirm_password = self.confirm_input.text()

        if password != confirm_password:
            self.message_label.setText("两次输入的密码不一致。")
            self.confirm_input.clear()
            self.confirm_input.setFocus()
            return

        ok, message = create_user(username, password)
        self.message_label.setText(message)
        if not ok:
            return

        self.registered_username = username
        QMessageBox.information(self, "注册成功", "新账号已创建，现在可以登录。")
        self.accept()
