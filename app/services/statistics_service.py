# app/services/statistics_service.py

from typing import Dict, List
from app.services.db_manager import (
    get_all_tasks,
    get_task_count,
    get_task_by_id,
    get_task_images
)


class StatisticsService:
    """统计服务（结果中心用）"""

    @staticmethod
    def get_recent_tasks(limit: int = 20) -> List[Dict]:
        """获取最近的任务列表"""
        return get_all_tasks(limit=limit)

    @staticmethod
    def get_task_detail(task_id: int) -> Dict:
        """获取任务详情"""
        task = get_task_by_id(task_id)
        if task:
            task['images'] = get_task_images(task_id)
        return task

    @staticmethod
    def get_total_count() -> int:
        """获取任务总数"""
        return get_task_count()
