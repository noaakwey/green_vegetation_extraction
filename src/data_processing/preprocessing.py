"""
Модуль предварительной обработки данных
"""

import numpy as np
from scipy import ndimage
from skimage import filters
from typing import Tuple

def apply_gaussian_filter(data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Применение гауссова фильтра для уменьшения шума

    Args:
        data (np.ndarray): Входные данные
        sigma (float): Стандартное отклонение фильтра

    Returns:
        np.ndarray: Обработанные данные
    """
    if len(data.shape) == 3:
        # Для RGB изображений фильтруем каждый канал отдельно
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[2]):
            filtered_data[:, :, i] = ndimage.gaussian_filter(data[:, :, i], sigma=sigma)
        return filtered_data
    else:
        return ndimage.gaussian_filter(data, sigma=sigma)

def enhance_contrast(data: np.ndarray, method: str = 'histogram') -> np.ndarray:
    """
    Улучшение контраста изображения

    Args:
        data (np.ndarray): Входные данные
        method (str): Метод улучшения контраста ('histogram', 'clahe')

    Returns:
        np.ndarray: Обработанные данные
    """
    if method == 'histogram':
        # Гистограммная эквализация
        if len(data.shape) == 3:
            # Для RGB изображений обрабатываем каждый канал
            enhanced_data = np.zeros_like(data)
            for i in range(data.shape[2]):
                enhanced_data[:, :, i] = filters.equalize_hist(data[:, :, i])
            return enhanced_data
        else:
            return filters.equalize_hist(data)
    else:
        return data

def remove_noise(data: np.ndarray, method: str = 'median') -> np.ndarray:
    """
    Удаление шума изображения

    Args:
        data (np.ndarray): Входные данные
        method (str): Метод удаления шума ('median', 'gaussian')

    Returns:
        np.ndarray: Обработанные данные
    """
    if method == 'median':
        # Медианный фильтр
        if len(data.shape) == 3:
            filtered_data = np.zeros_like(data)
            for i in range(data.shape[2]):
                filtered_data[:, :, i] = ndimage.median_filter(data[:, :, i], size=3)
            return filtered_data
        else:
            return ndimage.median_filter(data, size=3)
    else:
        return data

def normalize_rgb_channels(data: np.ndarray) -> np.ndarray:
    """
    Нормализация RGB каналов

    Args:
        data (np.ndarray): Входные данные

    Returns:
        np.ndarray: Нормализованные данные
    """
    # Приведение значений к диапазону 0-255
    if data.dtype != np.uint8:
        # Нормализация в диапазон 0-255
        normalized_data = np.zeros_like(data, dtype=np.uint8)
        for i in range(data.shape[2]):
            channel = data[:, :, i]
            normalized_data[:, :, i] = ((channel - channel.min()) /
                                      (channel.max() - channel.min()) * 255).astype(np.uint8)
        return normalized_data
    return data