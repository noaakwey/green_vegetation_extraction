"""
Модуль выделения объектов по геометрическим характеристикам
"""

import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import binary_opening, binary_closing
from typing import List, Tuple
import sys
import os

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classification.threshold_classification import threshold_classification

def extract_objects(binary_mask: np.ndarray,
                   min_area: int = 100,
                   max_area: int = 10000,
                   connectivity: int = 8) -> List:
    """
    Выделение объектов из бинарной маски с фильтрацией по площади

    Args:
        binary_mask (np.ndarray): Бинарная маска
        min_area (int): Минимальная площадь объекта
        max_area (int): Максимальная площадь объекта
        connectivity (int): Связность (4 или 8)

    Returns:
        List: Список объектов с геометрическими характеристиками
    """
    # Метки объектов
    labeled = label(binary_mask, connectivity=connectivity)

    # Получение свойств объектов
    regions = regionprops(labeled)

    # Фильтрация по площади
    valid_regions = []
    for region in regions:
        if min_area <= region.area <= max_area:
            valid_regions.append(region)

    return valid_regions

def filter_by_shape(regions: List, min_circularity: float = 0.5) -> List:
    """
    Фильтрация объектов по форме (круглость)

    Args:
        regions (List): Список объектов
        min_circularity (float): Минимальная круглость (0-1)

    Returns:
        List: Отфильтрованные объекты
    """
    filtered_regions = []
    for region in regions:
        # Вычисление круглости
        if region.perimeter > 0:
            circularity = 4 * np.pi * region.area / (region.perimeter ** 2)
            if circularity >= min_circularity:
                filtered_regions.append(region)
        else:
            # Если периметр равен 0, добавляем объект
            filtered_regions.append(region)

    return filtered_regions

def morphological_operations(binary_mask: np.ndarray,
                           opening_size: int = 3,
                           closing_size: int = 3) -> np.ndarray:
    """
    Морфологические операции для улучшения маски

    Args:
        binary_mask (np.ndarray): Бинарная маска
        opening_size (int): Размер для операции открытия
        closing_size (int): Размер для операции закрытия

    Returns:
        np.ndarray: Улучшенная маска
    """
    # Создание структурного элемента
    from skimage.morphology import disk

    # Операция открытия (удаление шума)
    if opening_size > 0:
        selem_open = disk(opening_size)
        opened_mask = binary_opening(binary_mask, selem=selem_open)
    else:
        opened_mask = binary_mask

    # Операция закрытия (заполнение дыр)
    if closing_size > 0:
        selem_close = disk(closing_size)
        closed_mask = binary_closing(opened_mask, selem=selem_close)
    else:
        closed_mask = opened_mask

    return closed_mask

def extract_vegetation_objects(rgb_data: np.ndarray,
                              min_area: int = 100,
                              max_area: int = 10000,
                              connectivity: int = 8) -> List:
    """
    Полный процесс выделения зеленых насаждений

    Args:
        rgb_data (np.ndarray): RGB данные изображения
        min_area (int): Минимальная площадь объекта
        max_area (int): Максимальная площадь объекта
        connectivity (int): Связность

    Returns:
        List: Список выделенных объектов
    """
    # Пороговая классификация
    binary_mask = threshold_classification(rgb_data)

    # Морфологические операции
    processed_mask = morphological_operations(binary_mask)

    # Выделение объектов
    objects = extract_objects(processed_mask, min_area, max_area, connectivity)

    return objects