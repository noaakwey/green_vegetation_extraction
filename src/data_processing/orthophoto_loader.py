"""
Модуль загрузки и предварительной обработки ортофотопланов
"""

import numpy as np
from PIL import Image
import os
from typing import Tuple, Optional

def load_orthophoto(file_path: str) -> np.ndarray:
    """
    Загрузка ортофотоплана из файла

    Args:
        file_path (str): Путь к файлу ортофотоплана

    Returns:
        np.ndarray: Массив изображения в формате (height, width, channels)
    """
    try:
        # Загружаем изображение с помощью PIL
        img = Image.open(file_path)
        img_array = np.array(img)

        # Обработка альфа-канала (если присутствует)
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            # Убираем альфа-канал, оставляем только RGB
            img_array = img_array[:, :, :3]

        return img_array

    except Exception as e:
        raise Exception(f"Ошибка загрузки ортофотоплана {file_path}: {str(e)}")

def save_orthophoto(data: np.ndarray, file_path: str):
    """
    Сохранение ортофотоплана в файл

    Args:
        data (np.ndarray): Данные изображения
        file_path (str): Путь для сохранения
    """
    try:
        # Преобразуем массив в изображение
        img = Image.fromarray(data.astype(np.uint8))
        img.save(file_path)
    except Exception as e:
        raise Exception(f"Ошибка сохранения ортофотоплана {file_path}: {str(e)}")

def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Нормализация данных (если требуется)

    Args:
        data (np.ndarray): Входные данные

    Returns:
        np.ndarray: Нормализованные данные
    """
    # Для 8-битных данных обычно нормализация не требуется
    # Но можно добавить проверку диапазона значений
    if data.dtype != np.uint8:
        # Нормализация в диапазон 0-255
        data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)

    return data

def preprocess_orthophoto(data: np.ndarray) -> np.ndarray:
    """
    Предварительная обработка ортофотоплана

    Args:
        data (np.ndarray): Входные данные

    Returns:
        np.ndarray: Обработанные данные
    """
    # Здесь можно добавить предварительную обработку:
    # - устранение шума
    # - усиление контраста
    # - нормализация
    # - фильтрация

    return data