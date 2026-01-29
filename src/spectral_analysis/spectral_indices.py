"""
Модуль спектрального анализа для выделения зеленых насаждений
"""

import numpy as np
from typing import Tuple, Dict

def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Вычисление индекса нормализованной разности растительности (NDVI)

    Args:
        red (np.ndarray): Канал красного света
        nir (np.ndarray): Канал ближнего инфракрасного света

    Returns:
        np.ndarray: NDVI матрица
    """
    # Для RGB данных NDVI не может быть напрямую вычислен, но можно использовать упрощенные индексы
    # Используем упрощенный NDVI с RGB каналами
    ndvi = (nir.astype(np.float32) - red.astype(np.float32)) / (nir.astype(np.float32) + red.astype(np.float32) + 0.001)
    return ndvi

def calculate_green_ratio(red: np.ndarray, green: np.ndarray) -> np.ndarray:
    """
    Вычисление отношения зеленого к красному каналу

    Args:
        red (np.ndarray): Канал красного света
        green (np.ndarray): Канал зеленого света

    Returns:
        np.ndarray: Соотношение G/R
    """
    ratio = green.astype(np.float32) / (red.astype(np.float32) + 0.001)
    return ratio

def calculate_green_blue_ratio(green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """
    Вычисление отношения зеленого к синему каналу

    Args:
        green (np.ndarray): Канал зеленого света
        blue (np.ndarray): Канал синего света

    Returns:
        np.ndarray: Соотношение G/B
    """
    ratio = green.astype(np.float32) / (blue.astype(np.float32) + 0.001)
    return ratio

def calculate_green_index(red: np.ndarray, green: np.ndarray) -> np.ndarray:
    """
    Вычисление индекса зелености

    Args:
        red (np.ndarray): Канал красного света
        green (np.ndarray): Канал зеленого света

    Returns:
        np.ndarray: Индекс зелености
    """
    # Упрощенный индекс зелености
    index = (green.astype(np.float32) - red.astype(np.float32)) / (green.astype(np.float32) + red.astype(np.float32) + 0.001)
    return index

def calculate_color_indices(rgb_data: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Вычисление всех цветовых индексов

    Args:
        rgb_data (np.ndarray): RGB данные изображения

    Returns:
        Dict[str, np.ndarray]: Словарь с цветовыми индексами
    """
    # Разделение каналов
    r = rgb_data[:, :, 0].astype(np.float32)
    g = rgb_data[:, :, 1].astype(np.float32)
    b = rgb_data[:, :, 2].astype(np.float32)

    # Вычисление индексов
    indices = {
        'green_red_ratio': calculate_green_ratio(r, g),
        'green_blue_ratio': calculate_green_blue_ratio(g, b),
        'green_index': calculate_green_index(r, g)
    }

    return indices