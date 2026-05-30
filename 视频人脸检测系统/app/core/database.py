import hashlib
import hmac
import sqlite3
from datetime import datetime
from pathlib import Path


DATABASE_PATH = Path("data/app.db")
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "123456"

# 获取数据库位置
def get_database_path() -> Path:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return DATABASE_PATH.resolve()

# 初始化数据库
def initialize_database() -> None:
    db_path = get_database_path()
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS detection_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                source_video TEXT NOT NULL,
                result_video TEXT NOT NULL,
                model_path TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS image_detection_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                source_image TEXT NOT NULL,
                result_image TEXT NOT NULL,
                model_path TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO users (username, password_hash, created_at)
            VALUES (?, ?, ?)
            """,
            (
                DEFAULT_USERNAME,
                hash_password(DEFAULT_PASSWORD),
                current_timestamp(),
            ),
        )
        conn.commit()

# 校验用户
def authenticate_user(username: str, password: str) -> bool:
    db_path = get_database_path()
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()

    if row is None:
        return False

    return hmac.compare_digest(row[0], hash_password(password))

# 注册用户
def create_user(username: str, password: str) -> tuple[bool, str]:
    clean_username = username.strip()
    if not clean_username:
        return False, "用户名不能为空。"
    if len(clean_username) < 3:
        return False, "用户名至少需要 3 个字符。"
    if not password:
        return False, "密码不能为空。"
    if len(password) < 6:
        return False, "密码至少需要 6 位。"

    db_path = get_database_path()
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO users (username, password_hash, created_at)
                VALUES (?, ?, ?)
                """,
                (
                    clean_username,
                    hash_password(password),
                    current_timestamp(),
                ),
            )
            conn.commit()
    except sqlite3.IntegrityError:
        return False, "用户名已存在，请更换一个用户名。"

    return True, "注册成功。"

# 保存视频记录
def save_detection_record(
    username: str,
    source_video: str,
    result_video: str,
    model_path: str,
) -> int:
    db_path = get_database_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO detection_records (
                username,
                source_video,
                result_video,
                model_path,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                username,
                source_video,
                result_video,
                model_path,
                current_timestamp(),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)

# 反回视频记录的总数
def count_detection_records() -> int:
    db_path = get_database_path()
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) FROM detection_records").fetchone()
    return int(row[0]) if row else 0

# 获取视频记录
def fetch_detection_records(limit: int = 100, username: str | None = None) -> list[dict]:
    db_path = get_database_path()
    query = (
        """
        SELECT id, username, source_video, result_video, model_path, created_at
        FROM detection_records
        """
    )
    params = []
    if username:
        query += " WHERE username = ?"
        params.append(username)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    rows.reverse()

    return [
        {
            "id": int(row[0]),
            "username": row[1],
            "source_video": row[2],
            "result_video": row[3],
            "model_path": row[4],
            "created_at": row[5],
        }
        for row in rows
    ]

# 删除视频记录
def delete_detection_record(record_id: int) -> bool:
    db_path = get_database_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            "DELETE FROM detection_records WHERE id = ?",
            (record_id,),
        )
        conn.commit()
        return cursor.rowcount > 0


def save_image_detection_record(
    username: str,
    source_image: str,
    result_image: str,
    model_path: str,
) -> int:
    db_path = get_database_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO image_detection_records (
                username,
                source_image,
                result_image,
                model_path,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                username,
                source_image,
                result_image,
                model_path,
                current_timestamp(),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def count_image_detection_records() -> int:
    db_path = get_database_path()
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) FROM image_detection_records").fetchone()
    return int(row[0]) if row else 0


def fetch_image_detection_records(limit: int = 100, username: str | None = None) -> list[dict]:
    db_path = get_database_path()
    query = (
        """
        SELECT id, username, source_image, result_image, model_path, created_at
        FROM image_detection_records
        """
    )
    params = []
    if username:
        query += " WHERE username = ?"
        params.append(username)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    rows.reverse()

    return [
        {
            "id": int(row[0]),
            "username": row[1],
            "source_image": row[2],
            "result_image": row[3],
            "model_path": row[4],
            "created_at": row[5],
        }
        for row in rows
    ]

def delete_image_detection_record(record_id: int) -> bool:
    db_path = get_database_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            "DELETE FROM image_detection_records WHERE id = ?",
            (record_id,),
        )
        conn.commit()
        return cursor.rowcount > 0

# 获取 creat_at 时间
def current_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 返回哈希密码
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()
