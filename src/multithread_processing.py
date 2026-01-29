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

from data_processing.orthophoto_loader import load_orthophoto, preprocess_orthophoto
from data_processing.preprocessing import apply_gaussian_filter, normalize_rgb_channels
from classification.threshold_classification import threshold_classification
from geometry_processing.object_extraction import extract_vegetation_objects
from geospatial_processing import load_polygon_shapefile, get_orthophoto_info

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

    # Настройки для работы с локальными файлами
    if local_directory is None:
        # Используем временную директорию в системной папке
        local_directory = os.path.join(os.path.expanduser("~"), "dask_scratch")
        os.makedirs(local_directory, exist_ok=True)

    # Создаем клиент с настройками
    client = Client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit='auto',
        local_directory=local_directory
    )

    print(f"Запущен Dask клиент с {n_workers} рабочими процессами")
    print(f"Используется локальная директория: {local_directory}")
    return client

@delayed
def process_orthophoto_chunk(orthophoto_path: str,
                           polygon_geom,
                           output_dir: str,
                           min_area: int = 100,
                           max_area: int = 10000) -> Dict:
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
        # Загрузка и предварительная обработка
        orthophoto = load_orthophoto(orthophoto_path)
        processed_orthophoto = preprocess_orthophoto(orthophoto)
        processed_orthophoto = apply_gaussian_filter(processed_orthophoto, sigma=1.0)
        processed_orthophoto = normalize_rgb_channels(processed_orthophoto)

        # Выделение зеленых насаждений
        vegetation_objects = extract_vegetation_objects(
            processed_orthophoto,
            min_area=min_area,
            max_area=max_area
        )

        # Возвращаем результаты
        return {
            'polygon': polygon_geom,
            'count': len(vegetation_objects),
            'objects': vegetation_objects,
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
            task = process_orthophoto_chunk(
                orthophoto_path,
                polygon,
                os.path.join(output_dir, f"polygon_{i}"),
                min_area,
                max_area
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

        # Создаем задачу для полной обработки
        task = process_orthophoto_chunk(
            orthophoto_path,
            None,  # Без ограничений по полигону
            output_dir,
            min_area,
            max_area
        )

        # Выполняем задачу
        result = dask.compute(task)[0]

        return {
            'total_objects': result['count'],
            'success': result['success'],
            'error': result['error']
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