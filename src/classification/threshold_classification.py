"""
Модуль пороговой классификации для выделения зеленых насаждений
"""

import numpy as np
from typing import Tuple, Dict
import sys
import os

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectral_analysis.spectral_indices import calculate_color_indices

def threshold_classification(rgb_data: np.ndarray,
                            green_ratio_threshold: float = 1.0,
                            green_index_threshold: float = 0.2,
                            min_area: int = 100) -> np.ndarray:
    """
    Пороговая классификация зеленых насаждений

    Args:
        rgb_data (np.ndarray): RGB данные изображения
        green_ratio_threshold (float): Пороговое значение соотношения G/R
        green_index_threshold (float): Пороговое значение индекса зелености
        min_area (int): Минимальная площадь объекта

    Returns:
        np.ndarray: Бинарная маска с выделенными зелеными насаждениями
    """
    # Вычисление цветовых индексов
    indices = calculate_color_indices(rgb_data)

    # Получаем индексы
    green_red_ratio = indices['green_red_ratio']
    green_blue_ratio = indices['green_blue_ratio']
    green_index = indices['green_index']

    # Создание бинарной маски по порогам
    mask = (green_red_ratio >= green_ratio_threshold) & \
           (green_blue_ratio >= green_ratio_threshold) & \
           (green_index >= green_index_threshold)

    # Преобразуем в uint8 формат
    binary_mask = mask.astype(np.uint8) * 255

    return binary_mask

def adaptive_threshold_classification(rgb_data: np.ndarray,
                                    local_window_size: int = 15) -> np.ndarray:
    """
    Адаптивная пороговая классификация с использованием локальных порогов

    Args:
        rgb_data (np.ndarray): RGB данные изображения
        local_window_size (int): Размер локального окна для адаптивного порога

    Returns:
        np.ndarray: Бинарная маска с выделенными зелеными насаждениями
    """
    # Для адаптивной классификации можно использовать более сложные методы
    # В данном случае используем простую реализацию

    # Вычисление цветовых индексов
    indices = calculate_color_indices(rgb_data)

    # Простая адаптивная классификация по средним значениям
    green_red_ratio = indices['green_red_ratio']
    green_blue_ratio = indices['green_blue_ratio']

    # Усреднение по локальным окнам
    from scipy import ndimage
    green_red_mean = ndimage.uniform_filter(green_red_ratio, size=local_window_size)
    green_blue_mean = ndimage.uniform_filter(green_blue_ratio, size=local_window_size)

    # Создание маски
    mask = (green_red_ratio >= green_red_mean * 0.8) & \
           (green_blue_ratio >= green_blue_mean * 0.8)

    binary_mask = mask.astype(np.uint8) * 255

    return binary_mask

def combined_classification(rgb_data: np.ndarray,
                           thresholds: Dict[str, float] = None) -> np.ndarray:
    """
    Комбинированная классификация с использованием нескольких порогов

    Args:
        rgb_data (np.ndarray): RGB данные изображения
        thresholds (Dict[str, float]): Словарь пороговых значений

    Returns:
        np.ndarray: Бинарная маска с выделенными зелеными насаждениями
    """
    if thresholds is None:
        thresholds = {
            'green_red_ratio': 1.0,
            'green_blue_ratio': 1.0,
            'green_index': 0.2
        }

    # Вычисление цветовых индексов
    indices = calculate_color_indices(rgb_data)

    # Создание маски с комбинированным условием
    mask = (indices['green_red_ratio'] >= thresholds['green_red_ratio']) & \
           (indices['green_blue_ratio'] >= thresholds['green_blue_ratio']) & \
           (indices['green_index'] >= thresholds['green_index'])

    binary_mask = mask.astype(np.uint8) * 255

    return binary_mask