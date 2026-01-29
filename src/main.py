"""
Основной скрипт проекта по выделению зеленых насаждений
"""

import numpy as np
import os
import sys
import argparse
from typing import List, Dict

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing.orthophoto_loader import load_orthophoto, preprocess_orthophoto
from data_processing.preprocessing import apply_gaussian_filter, normalize_rgb_channels
from classification.threshold_classification import threshold_classification, combined_classification
from geometry_processing.object_extraction import extract_vegetation_objects, morphological_operations
from output.export_results import export_to_geojson, export_to_shapefile, export_results_summary
from geospatial_processing import load_polygon_shapefile, process_polygons_in_orthophoto, get_orthophoto_info
from multithread_processing import process_orthophoto_parallel, process_orthophoto_full
import rasterio

def process_orthophoto_in_polygons(orthophoto_path: str,
                                  shapefile_path: str,
                                  output_dir: str,
                                  buffer_meters: float = 0.0,
                                  n_workers: int = None) -> None:
    """
    Обработка ортофотоплана по полигонам из Shapefile с использованием многопоточности

    Args:
        orthophoto_path (str): Путь к ортофотоплану
        shapefile_path (str): Путь к Shapefile с полигонами
        output_dir (str): Директория для вывода результатов
        buffer_meters (float): Буфер вокруг полигонов
        n_workers (int): Количество рабочих процессов
    """
    print(f"Обработка ортофотоплана по полигонам: {orthophoto_path}")
    print(f"Формат данных: {get_orthophoto_info(orthophoto_path)}")

    # Используем многопоточную обработку
    results = process_orthophoto_parallel(
        orthophoto_path,
        shapefile_path,
        output_dir,
        n_workers=n_workers,
        local_directory=os.path.join(os.path.expanduser("~"), "dask_scratch")
    )

    print("Обработка завершена")
    print(f"Результаты: {results}")

def process_orthophoto(input_path: str,
                       output_dir: str,
                       min_area: int = 100,
                       max_area: int = 10000,
                       n_workers: int = None) -> None:
    """
    Основной процесс обработки ортофотоплана

    Args:
        input_path (str): Путь к входному файлу ортофотоплана
        output_dir (str): Директория для вывода результатов
        min_area (int): Минимальная площадь объекта
        max_area (int): Максимальная площадь объекта
        n_workers (int): Количество рабочих процессов
    """
    print(f"Загрузка ортофотоплана: {input_path}")

    # Устанавливаем лимиты для больших файлов
    rasterio.env.Env(
        GDAL_MAX_DATASET_OPENED=1000,
        GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
        GDAL_MAX_OPENED_DATASET=1000,
        GDAL_DISABLE_OPEN_OF_LARGE_FILE=0,
        VSI_CACHE=1,
        VSI_CACHE_SIZE=1073741824  # 1GB cache
    )

    # Для больших файлов используем простую последовательную обработку
    # вместо Dask для избежания проблем с памятью
    try:
        # Проверяем размер файла
        try:
            file_size = os.path.getsize(input_path)
            print(f"Размер файла: {file_size / (1024**3):.2f} GB")
        except Exception as e:
            print(f"Не удалось получить размер файла: {e}")

        # Используем простую последовательную обработку
        results = process_orthophoto_simple(
            input_path,
            output_dir,
            min_area=min_area,
            max_area=max_area
        )

        print(f"Обработка завершена. Найдено {results['total_objects']} объектов")
        if not results['success']:
            print(f"Ошибка: {results['error']}")
    except Exception as e:
        print(f"Ошибка при обработке: {str(e)}")
        import traceback
        traceback.print_exc()


def process_orthophoto_simple(orthophoto_path: str,
                              output_dir: str,
                              min_area: int = 100,
                              max_area: int = 10000) -> Dict:
    """
    Простая последовательная обработка ортофотоплана без Dask

    Args:
        orthophoto_path (str): Путь к ортофотоплану
        output_dir (str): Директория для вывода результатов
        min_area (int): Минимальная площадь объекта
        max_area (int): Максимальная площадь объекта

    Returns:
        Dict: Результаты обработки
    """
    try:
        # Загружаем ортофотоплан
        print("Загрузка ортофотоплана...")
        orthophoto = load_orthophoto(orthophoto_path)
        print(f"Ортофотоплан загружен. Размер: {orthophoto.shape}")

        # Предварительная обработка
        print("Предварительная обработка...")
        processed_orthophoto = preprocess_orthophoto(orthophoto)
        processed_orthophoto = apply_gaussian_filter(processed_orthophoto, sigma=1.0)
        processed_orthophoto = normalize_rgb_channels(processed_orthophoto)

        # Выделение зеленых насаждений
        print("Выделение зеленых насаждений...")
        vegetation_objects = extract_vegetation_objects(
            processed_orthophoto,
            min_area=min_area,
            max_area=max_area
        )

        # Сохранение результатов
        print("Сохранение результатов...")
        os.makedirs(output_dir, exist_ok=True)

        # Сохраняем результаты в файлы
        results_summary = {
            'total_objects': len(vegetation_objects),
            'objects': vegetation_objects,
            'success': True,
            'error': None
        }

        # Просто сохраняем информацию о найденных объектах
        summary_file = os.path.join(output_dir, "results_summary.json")
        import json
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, ensure_ascii=False, indent=2)

        return results_summary

    except Exception as e:
        print(f"Ошибка при простой обработке: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'total_objects': 0,
            'success': False,
            'error': str(e)
        }

def main():
    """
    Основная функция
    """
    # Создаем парсер аргументов
    parser = argparse.ArgumentParser(description='Обработка ортофотопланов для выделения зеленых насаждений')
    parser.add_argument('orthophoto_path', help='Путь к ортофотоплану')
    parser.add_argument('--shapefile', help='Путь к Shapefile с полигонами (необязательно)')
    parser.add_argument('--output', default='data/output/', help='Директория для вывода результатов')
    parser.add_argument('--workers', type=int, default=128, help='Количество рабочих процессов (по умолчанию 128)')
    parser.add_argument('--min_area', type=int, default=100, help='Минимальная площадь объекта')
    parser.add_argument('--max_area', type=int, default=10000, help='Максимальная площадь объекта')

    args = parser.parse_args()

    # Проверяем существование ортофотоплана
    try:
        if not os.path.exists(args.orthophoto_path):
            print(f"Ошибка: Ортофотоплан не найден по пути {args.orthophoto_path}")
            return
    except Exception as e:
        print(f"Ошибка проверки пути к ортофотоплану: {str(e)}")
        # Попробуем обработать путь как байтовую строку
        try:
            # Пробуем открыть файл для проверки существования
            with open(args.orthophoto_path, 'rb') as f:
                pass
            print("Файл существует, но возникли проблемы с кодировкой пути")
        except Exception:
            print("Файл не найден")
            return

    # Создаем директории для вывода
    try:
        os.makedirs(args.output, exist_ok=True)
    except Exception as e:
        print(f"Ошибка создания директории вывода: {str(e)}")
        return

    print(f"Обработка ортофотоплана: {args.orthophoto_path}")
    print(f"Используем {args.workers} рабочих процессов")

    # Проверяем наличие Shapefile
    if args.shapefile:
        try:
            if not os.path.exists(args.shapefile):
                print(f"Shapefile не найден по пути {args.shapefile}")
                args.shapefile = None
        except Exception as e:
            print(f"Ошибка проверки пути к Shapefile: {str(e)}")
            args.shapefile = None

    if args.shapefile:
        print("Обнаружен Shapefile с полигонами - будет использован для ограничения области обработки")
        print(f"Используем {args.workers} рабочих процессов для параллельной обработки")

        # Обработка по полигонам с использованием многопоточности
        try:
            process_orthophoto_in_polygons(
                args.orthophoto_path,
                args.shapefile,
                args.output,
                n_workers=args.workers,
                min_area=args.min_area,
                max_area=args.max_area
            )
        except Exception as e:
            print(f"Ошибка обработки по полигонам: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("Shapefile не указан - будет обработан весь ортофотоплан")
        print(f"Используем {args.workers} рабочих процессов для параллельной обработки")

        # Обработка всего ортофотоплана с использованием многопоточности
        try:
            process_orthophoto(
                args.orthophoto_path,
                args.output,
                n_workers=args.workers,
                min_area=args.min_area,
                max_area=args.max_area
            )
        except Exception as e:
            print(f"Ошибка полной обработки: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()