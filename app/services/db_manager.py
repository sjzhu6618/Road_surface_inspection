import json
from pathlib import Path
from typing import List, Optional

import pymysql


def get_db_connection():
    return pymysql.connect(
        host='localhost',
        port=3306,
        user='root',
        password='3.15926',
        database='road_damage_db',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor,
    )


def init_database():
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql_create_tasks = """
                CREATE TABLE IF NOT EXISTS detection_tasks (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    task_name VARCHAR(255) COMMENT '任务名称',
                    source_path VARCHAR(500) COMMENT '源文件路径',
                    task_type VARCHAR(50) COMMENT '任务类型(单图/视频/文件夹)',
                    total_detections INT DEFAULT 0 COMMENT '检测目标总数',
                    total_area_m2 FLOAT DEFAULT 0 COMMENT '总受损面积',
                    total_volume_m3 FLOAT DEFAULT 0 COMMENT '总体积',
                    avg_depth_cm FLOAT DEFAULT 0 COMMENT '平均深度',
                    severe_count INT DEFAULT 0 COMMENT '重度数量',
                    moderate_count INT DEFAULT 0 COMMENT '中度数量',
                    mild_count INT DEFAULT 0 COMMENT '轻度数量',
                    summary_json JSON COMMENT '完整 summary.json',
                    dashboard_json JSON COMMENT '完整 dashboard_data.json',
                    result_video_path VARCHAR(500) COMMENT '结果视频路径',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_created (created_at),
                    INDEX idx_task_name (task_name),
                    INDEX idx_source (source_path)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='路面损伤检测任务表';
            """

            sql_create_images = """
                CREATE TABLE IF NOT EXISTS result_images (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    task_id INT COMMENT '关联任务ID',
                    image_type VARCHAR(50) COMMENT '图片类型(pred/depth/annotated)',
                    image_name VARCHAR(255) COMMENT '图片文件名',
                    image_path VARCHAR(500) COMMENT '文件路径',
                    frame_index INT COMMENT '帧索引，视频用',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (task_id) REFERENCES detection_tasks(id) ON DELETE CASCADE,
                    INDEX idx_task (task_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='结果图片表';
            """

            cursor.execute(sql_create_tasks)
            cursor.execute(sql_create_images)
        conn.commit()
    finally:
        conn.close()


def save_detection_task(
    summary_data: dict,
    dashboard_data: list,
    preview_paths: dict = None,
    video_path: str = None,
    image_records: list = None,
) -> int:
    conn = get_db_connection()
    try:
        source_path = summary_data.get('source', '')
        if not source_path:
            return -1

        total_detections = len(dashboard_data) if isinstance(dashboard_data, list) else 0
        severe_count = sum(1 for item in dashboard_data if item.get('status') == '重度')
        moderate_count = sum(1 for item in dashboard_data if item.get('status') == '中度')
        mild_count = sum(1 for item in dashboard_data if item.get('status') == '轻度')
        total_area = sum(item.get('metrics', {}).get('area_m2', 0) for item in dashboard_data)
        total_volume = sum(item.get('metrics', {}).get('volume_m3', 0) for item in dashboard_data)
        depths = [item.get('metrics', {}).get('max_depth_cm', 0) for item in dashboard_data]
        depths = [d for d in depths if d > 0]
        avg_depth = sum(depths) / len(depths) if depths else 0

        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id
                FROM detection_tasks
                WHERE source_path = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (source_path,),
            )
            existing = cursor.fetchone()

            task_name = summary_data.get('task_name', '未命名任务')
            task_type = summary_data.get('类型', '单图')

            if existing:
                task_id = existing['id']
                cursor.execute(
                    """
                    UPDATE detection_tasks
                    SET task_name = %s,
                        task_type = %s,
                        total_detections = %s,
                        total_area_m2 = %s,
                        total_volume_m3 = %s,
                        avg_depth_cm = %s,
                        severe_count = %s,
                        moderate_count = %s,
                        mild_count = %s,
                        summary_json = %s,
                        dashboard_json = %s,
                        result_video_path = %s,
                        created_at = NOW()
                    WHERE id = %s
                    """,
                    (
                        task_name,
                        task_type,
                        total_detections,
                        round(total_area, 4),
                        round(total_volume, 5),
                        round(avg_depth, 2),
                        severe_count,
                        moderate_count,
                        mild_count,
                        json.dumps(summary_data, ensure_ascii=False),
                        json.dumps(dashboard_data, ensure_ascii=False),
                        video_path,
                        task_id,
                    ),
                )
                cursor.execute("DELETE FROM result_images WHERE task_id = %s", (task_id,))
            else:
                cursor.execute(
                    """
                    INSERT INTO detection_tasks (
                        task_name, source_path, task_type,
                        total_detections, total_area_m2, total_volume_m3,
                        avg_depth_cm, severe_count, moderate_count, mild_count,
                        summary_json, dashboard_json, result_video_path
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        task_name,
                        source_path,
                        task_type,
                        total_detections,
                        round(total_area, 4),
                        round(total_volume, 5),
                        round(avg_depth, 2),
                        severe_count,
                        moderate_count,
                        mild_count,
                        json.dumps(summary_data, ensure_ascii=False),
                        json.dumps(dashboard_data, ensure_ascii=False),
                        video_path,
                    ),
                )
                task_id = cursor.lastrowid

            if image_records:
                for item in image_records:
                    if not isinstance(item, dict):
                        continue
                    image_path = item.get('image_path') or item.get('path')
                    image_type = item.get('image_type') or item.get('type') or 'pred'
                    if image_path:
                        _save_image_record(cursor, task_id, image_path, image_type)
            elif preview_paths:
                for image_type, image_path in preview_paths.items():
                    if image_path:
                        _save_image_record(cursor, task_id, image_path, image_type)

        conn.commit()
        return task_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _save_image_record(cursor, task_id: int, image_path: str, image_type: str):
    image_name = Path(image_path).name
    cursor.execute(
        """
        INSERT INTO result_images (task_id, image_type, image_name, image_path)
        VALUES (%s, %s, %s, %s)
        """,
        (task_id, image_type, image_name, str(image_path)),
    )


def get_task_by_id(task_id: int) -> Optional[dict]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM detection_tasks WHERE id = %s", (task_id,))
            task = cursor.fetchone()
            if task:
                task['summary_data'] = json.loads(task['summary_json']) if task['summary_json'] else {}
                task['dashboard_data'] = json.loads(task['dashboard_json']) if task['dashboard_json'] else []
            return task
    finally:
        conn.close()


def get_all_tasks(limit: int = 100, offset: int = 0) -> List[dict]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id,
                       task_name,
                       source_path,
                       task_type,
                       total_detections,
                       total_area_m2,
                       total_volume_m3,
                       avg_depth_cm,
                       severe_count,
                       moderate_count,
                       mild_count,
                       result_video_path,
                       created_at
                FROM detection_tasks
                ORDER BY created_at DESC
                LIMIT %s
                OFFSET %s
                """,
                (limit, offset),
            )
            return cursor.fetchall()
    finally:
        conn.close()


def get_task_count() -> int:
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS count FROM detection_tasks")
            result = cursor.fetchone()
            return result['count'] if result else 0
    finally:
        conn.close()


def get_task_images(task_id: int) -> List[dict]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, image_type, image_name, image_path
                FROM result_images
                WHERE task_id = %s
                ORDER BY image_type, image_name
                """,
                (task_id,),
            )
            return cursor.fetchall()
    finally:
        conn.close()


def delete_task(task_id: int):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM detection_tasks WHERE id = %s", (task_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
