"""
Модуль многопоточной обработки ортофотопланов
"""

import dask
from dask import delayed
import dask.distributed
from dask.distributed import Client
import numpy as np
import os
from typing import List, Dict, Tuple
import sys

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Оптимизация для высоконагруженных систем
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from data_processing.orthophoto_loader import load_orthophoto, preprocess_orthophoto
from data_processing.preprocessing import apply_gaussian_filter, normalize_rgb_channels
from classification.threshold_classification import threshold_classification
from geometry_processing.object_extraction import extract_vegetation_objects
from geospatial_processing import load_polygon_shapefile, get_orthophoto_info
from rasterio.windows import Window

def setup_dask_client(n_workers: int = None, threads_per_worker: int = 1, local_directory: str = None) -> Client:
    """
    Настройка Dask клиента

    Args:
        n_workers (int): Количество рабочих процессов (по умолчанию - все ядра)
        threads_per_worker (int): Количество потоков на рабочий процесс
        local_directory (str): Локальная директория для scratch файлов

    Returns:
        Client: Dask клиент
    """
    if n_workers is None:
        # Используем все доступные ядра
        import multiprocessing
        n_workers = multiprocessing.cpu_count()

    # Оптимизация для 128 ядер на Windows
    # На Windows 'spawn' используется по умолчанию для multiprocessing в Python 3.8+
    # Для CPU-bound задач лучше 1 поток на рабочий процесс, чтобы избежать GIL
    
    # Настройки для работы с локальными файлами
    if local_directory is None:
        # Используем временную директорию в системной папке
        local_directory = os.path.join(os.path.expanduser("~"), "dask_scratch")
        os.makedirs(local_directory, exist_ok=True)

    # Создаем клиент с настройками
    # memory_limit='auto' на 1TB RAM - это отлично
    client = Client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit='auto',
        local_directory=local_directory
    )

    print(f"Запущен Dask клиент с {n_workers} рабочими процессами")
    print(f"Используется локальная директория: {local_directory}")
    return client

def get_tiles_windows(orthophoto_path: str, tile_size: int = 4096) -> List[Window]:
    """
    Генератор окон для тайловой обработки всего растра
    """
    with rasterio.open(orthophoto_path) as src:
        h, w = src.height, src.width
        windows = []
        for i in range(0, h, tile_size):
            for j in range(0, w, tile_size):
                window = Window(j, i, min(tile_size, w - j), min(tile_size, h - i))
                windows.append(window)
        return windows

@delayed
def process_orthophoto_chunk(orthophoto_path: str,
                           polygon_geom = None,
                           output_dir: str = None,
                           min_area: int = 100,
                           max_area: int = 10000,
                           window: Window = None) -> Dict:
    """
    Обработка одного фрагмента ортофотоплана

    Args:
        orthophoto_path (str): Путь к ортофотоплану
        polygon_geom: Геометрия полигона
        output_dir (str): Директория для вывода
        min_area (int): Минимальная площадь объекта
        max_area (int): Максимальная площадь объекта

    Returns:
        Dict: Результаты обработки
    """
    try:
        # Устанавливаем лимиты для больших файлов в каждом потоке
        import rasterio
        rasterio.env.Env(
            GDAL_MAX_DATASET_OPENED=1000,
            GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
            GDAL_MAX_OPENED_DATASET=1000,
            GDAL_DISABLE_OPEN_OF_LARGE_FILE=0,
            VSI_CACHE=1,
            VSI_CACHE_SIZE=1073741824,  # 1GB cache
            GDAL_CACHEMAX=1024 # 1GB GDAL cache
        )

        # Если окно не указано, но есть полигон, вычисляем окно из полигона
        if window is None and polygon_geom is not None:
            from rasterio.windows import from_bounds
            with rasterio.open(orthophoto_path) as src:
                bounds = polygon_geom.bounds
                window = from_bounds(*bounds, src.transform)
                # Округляем до ближайших пикселей, чтобы избежать проблем с выравниванием
                window = window.round_shape().round_offsets()

        # Загрузка и предварительная обработка (с использованием окна)
        orthophoto = load_orthophoto(orthophoto_path, window=window)
        processed_orthophoto = preprocess_orthophoto(orthophoto)
        processed_orthophoto = apply_gaussian_filter(processed_orthophoto, sigma=1.0)
        processed_orthophoto = normalize_rgb_channels(processed_orthophoto)

        # Выделение зеленых насаждений
        vegetation_objects = extract_vegetation_objects(
            processed_orthophoto,
            min_area=min_area,
            max_area=max_area
        )

        # Добавляем смещение окна к координатам объектов
        offset_x = window.col_off if window else 0
        offset_y = window.row_off if window else 0
        
        # Конвертируем RegionProperties в сериализуемые словари с глобальными координатами
        serializable_objects = []
        for reg in vegetation_objects:
            obj_data = {
                'area': reg.area,
                'bbox': [reg.bbox[0] + offset_y, reg.bbox[1] + offset_x, 
                         reg.bbox[2] + offset_y, reg.bbox[3] + offset_x],
                'centroid': [reg.centroid[0] + offset_y, reg.centroid[1] + offset_x],
                # Можно добавить еще свойства по необходимости
            }
            serializable_objects.append(obj_data)

        # Возвращаем результаты
        return {
            'polygon': polygon_geom,
            'count': len(serializable_objects),
            'objects': serializable_objects,
            'success': True,
            'error': None
        }

    except Exception as e:
        # Добавим дополнительную обработку ошибок кодировки
        error_msg = str(e)
        if "codec can't decode" in error_msg or "invalid continuation byte" in error_msg:
            print(f"Ошибка кодировки в пути к файлу: {orthophoto_path}")
            print(f"Детали ошибки: {error_msg}")
            # Попробуем работать с путем как байтовой строкой
            try:
                import pathlib
                path = pathlib.Path(orthophoto_path)
                if path.exists():
                    print("Файл существует, но есть проблемы с кодировкой пути")
                    return {
                        'polygon': polygon_geom,
                        'count': 0,
                        'objects': [],
                        'success': False,
                        'error': f"Проблема с кодировкой пути: {error_msg}"
                    }
            except Exception:
                pass
        return {
            'polygon': polygon_geom,
            'count': 0,
            'objects': [],
            'success': False,
            'error': error_msg
        }

def process_orthophoto_parallel(orthophoto_path: str,
                              shapefile_path: str,
                              output_dir: str,
                              n_workers: int = None,
                              min_area: int = 100,
                              max_area: int = 10000,
                              local_directory: str = None) -> Dict:
    """
    Параллельная обработка ортофотоплана по полигонам

    Args:
        orthophoto_path (str): Путь к ортофотоплану
        shapefile_path (str): Путь к Shapefile с полигонами
        output_dir (str): Директория для вывода
        n_workers (int): Количество рабочих процессов
        min_area (int): Минимальная площадь объекта
        max_area (int): Максимальная площадь объекта
        local_directory (str): Локальная директория для scratch файлов

    Returns:
        Dict: Результаты обработки
    """
    # Настройка Dask клиента
    client = setup_dask_client(n_workers, local_directory=local_directory)

    try:
        # Загрузка полигонов
        gdf = load_polygon_shapefile(shapefile_path)
        polygons = [geom for geom in gdf.geometry if geom is not None]

        print(f"Начинаем обработку {len(polygons)} полигонов")

        # Создаем задачи для параллельной обработки
        tasks = []
        for i, polygon in enumerate(polygons):
            # Для каждого полигона можно было бы тоже делать тайлинг, 
            # но пока оставим как один чанк, так как полигоны обычно небольшие.
            # Если полигон большой, он будет обработан целиком в одном чанке.
            task = process_orthophoto_chunk(
                orthophoto_path,
                polygon_geom=polygon,
                output_dir=os.path.join(output_dir, f"polygon_{i}"),
                min_area=min_area,
                max_area=max_area
            )
            tasks.append(task)

        # Выполняем задачи параллельно
        results = dask.compute(*tasks)

        # Собираем результаты
        total_objects = sum(result['count'] for result in results if result['success'])

        print(f"Обработка завершена. Найдено {total_objects} объектов")

        return {
            'total_polygons': len(polygons),
            'total_objects': total_objects,
            'results': results,
            'success': True
        }

    except Exception as e:
        print(f"Ошибка параллельной обработки: {e}")
        return {
            'total_polygons': 0,
            'total_objects': 0,
            'results': [],
            'success': False,
            'error': str(e)
        }
    finally:
        # Закрываем клиент
        client.close()

def process_orthophoto_full(orthophoto_path: str,
                          output_dir: str,
                          n_workers: int = None,
                          min_area: int = 100,
                          max_area: int = 10000,
                          local_directory: str = None) -> Dict:
    """
    Полная обработка ортофотоплана (без ограничений по полигонам)

    Args:
        orthophoto_path (str): Путь к ортофотоплану
        output_dir (str): Директория для вывода
        n_workers (int): Количество рабочих процессов
        min_area (int): Минимальная площадь объекта
        max_area (int): Максимальная площадь объекта
        local_directory (str): Локальная директория для scratch файлов

    Returns:
        Dict: Результаты обработки
    """
    # Настройка Dask клиента
    client = setup_dask_client(n_workers, local_directory=local_directory)

    try:
        # Получаем информацию об ортофотоплане
        info = get_orthophoto_info(orthophoto_path)
        print(f"Обработка полного ортофотоплана: {info}")

        # Генерируем тайлы для всего растра
        # Размер тайла 4096x4096 - хороший баланс между параллелизмом и оверхедом
        windows = get_tiles_windows(orthophoto_path, tile_size=4096)
        print(f"Растр разбит на {len(windows)} тайлов")

        # Создаем задачи для каждого тайла
        tasks = []
        for i, window in enumerate(windows):
            task = process_orthophoto_chunk(
                orthophoto_path,
                output_dir=output_dir,
                min_area=min_area,
                max_area=max_area,
                window=window
            )
            tasks.append(task)

        # Выполняем задачи параллельно
        results = dask.compute(*tasks)

        # Собираем результаты
        total_objects = sum(result['count'] for result in results if result['success'])

        return {
            'total_objects': total_objects,
            'success': True,
            'error': None
        }

    except Exception as e:
        print(f"Ошибка полной обработки: {e}")
        return {
            'total_objects': 0,
            'success': False,
            'error': str(e)
        }
    finally:
        # Закрываем клиент
        client.close()

# Пример использования:
if __name__ == "__main__":
    # Пример использования
    print("Модуль многопоточной обработки готов к использованию")