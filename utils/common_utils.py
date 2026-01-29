"""
Вспомогательные общие функции проекта
"""

import numpy as np
import os
from typing import List, Tuple

def validate_file_path(file_path: str) -> bool:
    """
    Проверка существования файла

    Args:
        file_path (str): Путь к файлу

    Returns:
        bool: True если файл существует
    """
    return os.path.exists(file_path) and os.path.isfile(file_path)

def validate_directory_path(dir_path: str) -> bool:
    """
    Проверка существования директории

    Args:
        dir_path (str): Путь к директории

    Returns:
        bool: True если директория существует
    """
    return os.path.exists(dir_path) and os.path.isdir(dir_path)

def get_file_list(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Получение списка файлов в директории с определенными расширениями

    Args:
        directory (str): Путь к директории
        extensions (List[str]): Список допустимых расширений

    Returns:
        List[str]: Список файлов
    """
    if not validate_directory_path(directory):
        return []

    files = []
    for file in os.listdir(directory):
        if extensions:
            if any(file.lower().endswith(ext.lower()) for ext in extensions):
                files.append(file)
        else:
            files.append(file)

    return files

def calculate_statistics(data: np.ndarray) -> dict:
    """
    Вычисление статистических характеристик массива

    Args:
        data (np.ndarray): Входные данные

    Returns:
        dict: Словарь со статистиками
    """
    stats = {
        'min': np.min(data),
        'max': np.max(data),
        'mean': np.mean(data),
        'std': np.std(data),
        'shape': data.shape
    }

    return stats

def create_output_directories(output_dir: str) -> None:
    """
    Создание необходимых директорий для вывода результатов

    Args:
        output_dir (str): Путь к директории вывода
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'geojson'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'shapefile'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)