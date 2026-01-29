"""
Модуль загрузки и предварительной обработки ортофотопланов
"""

import numpy as np
from PIL import Image
import os
from typing import Tuple, Optional
import rasterio
from rasterio.windows import Window

def load_orthophoto(file_path: str, window: Optional[Window] = None) -> np.ndarray:
    """
    Загрузка ортофотоплана из файла

    Args:
        file_path (str): Путь к файлу ортофотоплана
        window (Optional[Window]): Окно для чтения части изображения (для больших файлов)

    Returns:
        np.ndarray: Массив изображения в формате (height, width, channels)
    """
    try:
        # Проверяем размер файла
        file_size = os.path.getsize(file_path)
        # Если указано окно или файл очень большой, используем rasterio напрямую
        if window is not None or file_size > 1000000000:  # 1GB
            # Используем rasterio для больших файлов с настройками энкодинга
            with rasterio.Env(GDAL_FILENAME_IS_UTF8='YES', SHAPE_ENCODING='utf-8'):
                with rasterio.open(file_path) as src:
                    # Читаем каналы (с окном, если оно указано)
                    img_array = src.read(window=window)
                    # Переворачиваем ось для правильного формата (height, width, channels)
                    img_array = np.transpose(img_array, (1, 2, 0))
        else:
            # Для маленьких файлов используем PIL
            # Обрабатываем возможные проблемы с кодировкой пути
            try:
                img = Image.open(file_path)
                img_array = np.array(img)
            except (UnicodeDecodeError, Exception) as e:
                # Если есть проблемы с кодировкой, пробуем использовать pathlib
                try:
                    import pathlib
                    path = pathlib.Path(file_path)
                    if path.exists():
                        img = Image.open(str(path))
                        img_array = np.array(img)
                    else:
                        raise Exception(f"Файл не найден: {file_path}")
                except Exception:
                    # Если всё ещё не получилось, пробуем открыть как байтовый файл
                    try:
                        # Пробуем открыть файл в бинарном режиме и преобразовать
                        with open(file_path, 'rb') as f:
                            # Если это не изображение, возьмём просто байты
                            pass
                        # Если мы дошли до сюда, попробуем использовать rasterio напрямую
                        import rasterio as rio
                        with rio.Env(GDAL_FILENAME_IS_UTF8='YES', SHAPE_ENCODING='utf-8'):
                            with rio.open(file_path) as src:
                                img_array = src.read()
                                img_array = np.transpose(img_array, (1, 2, 0))
                    except Exception:
                        raise Exception(f"Не удалось загрузить файл: {file_path}")

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