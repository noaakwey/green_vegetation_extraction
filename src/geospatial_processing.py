"""
Модуль обработки геопространственных данных
"""

import geopandas as gpd
import numpy as np
from rasterio.mask import mask
from rasterio.features import geometry_mask
import rasterio
from shapely.geometry import box
from typing import List, Tuple, Dict
import os

def load_polygon_shapefile(shapefile_path: str) -> gpd.GeoDataFrame:
    """
    Загрузка полигональных данных из Shapefile

    Args:
        shapefile_path (str): Путь к Shapefile

    Returns:
        gpd.GeoDataFrame: Геоданные с полигонами
    """
    try:
        gdf = gpd.read_file(shapefile_path)
        return gdf
    except Exception as e:
        raise Exception(f"Ошибка загрузки Shapefile {shapefile_path}: {str(e)}")

def extract_polygon_geometries(gdf: gpd.GeoDataFrame) -> List:
    """
    Извлечение геометрий полигонов из GeoDataFrame

    Args:
        gdf (gpd.GeoDataFrame): Геоданные

    Returns:
        List: Список геометрий полигонов
    """
    return [geom for geom in gdf.geometry if geom is not None]

def get_polygon_bounds(geom) -> Tuple[float, float, float, float]:
    """
    Получение границ полигона

    Args:
        geom: Геометрия полигона

    Returns:
        Tuple[float, float, float, float]: Границы (minx, miny, maxx, maxy)
    """
    bounds = geom.bounds
    return (bounds[0], bounds[1], bounds[2], bounds[3])

def clip_orthophoto_to_polygon(orthophoto_path: str,
                              polygon_geom,
                              buffer_meters: float = 0.0) -> Tuple[np.ndarray, dict]:
    """
    Обрезка ортофотоплана по полигону

    Args:
        orthophoto_path (str): Путь к ортофотоплану
        polygon_geom: Геометрия полигона
        buffer_meters (float): Буфер в метрах

    Returns:
        Tuple[np.ndarray, dict]: Обрезанные данные и трансформация
    """
    try:
        with rasterio.open(orthophoto_path) as src:
            # Создаем буфер если нужно
            if buffer_meters > 0:
                buffered_geom = polygon_geom.buffer(buffer_meters)
            else:
                buffered_geom = polygon_geom

            # Получаем границы полигона
            bounds = get_polygon_bounds(buffered_geom)

            # Выполняем маскирование
            out_image, out_transform = mask(src, [buffered_geom], crop=True)

            # Получаем данные
            clipped_data = out_image

            return clipped_data, out_transform

    except Exception as e:
        raise Exception(f"Ошибка обрезки ортофотоплана: {str(e)}")

def process_polygons_in_orthophoto(orthophoto_path: str,
                                  shapefile_path: str,
                                  output_dir: str,
                                  buffer_meters: float = 0.0) -> Dict[str, int]:
    """
    Обработка ортофотоплана по полигонам из Shapefile

    Args:
        orthophoto_path (str): Путь к ортофотоплану
        shapefile_path (str): Путь к Shapefile с полигонами
        output_dir (str): Директория для вывода результатов
        buffer_meters (float): Буфер вокруг полигонов

    Returns:
        Dict[str, int]: Словарь с количеством найденных объектов по полигонам
    """
    # Загружаем полигоны
    gdf = load_polygon_shapefile(shapefile_path)

    # Извлекаем геометрии
    polygons = extract_polygon_geometries(gdf)

    results = {}

    # Обрабатываем каждый полигон
    for i, polygon in enumerate(polygons):
        print(f"Обработка полигона {i+1}/{len(polygons)}")

        try:
            # Обрезаем ортофотоплан по полигону
            clipped_data, transform = clip_orthophoto_to_polygon(orthophoto_path, polygon, buffer_meters)

            # Создаем уникальное имя для результата
            polygon_id = f"polygon_{i}"

            # Сохраняем обрезанный фрагмент для дальнейшей обработки
            polygon_output_dir = os.path.join(output_dir, polygon_id)
            os.makedirs(polygon_output_dir, exist_ok=True)

            # Здесь можно добавить дальнейшую обработку
            # Для примера, просто сохраняем информацию
            results[polygon_id] = {
                'area': polygon.area,
                'bounds': get_polygon_bounds(polygon),
                'processed': True
            }

        except Exception as e:
            print(f"Ошибка обработки полигона {i}: {str(e)}")
            results[f"polygon_{i}"] = {
                'error': str(e),
                'processed': False
            }

    return results

def get_orthophoto_info(orthophoto_path: str) -> Dict:
    """
    Получение информации об ортофотоплане

    Args:
        orthophoto_path (str): Путь к ортофотоплану

    Returns:
        Dict: Информация об ортофотоплане
    """
    try:
        with rasterio.open(orthophoto_path) as src:
            info = {
                'width': src.width,
                'height': src.height,
                'count': src.count,
                'dtype': src.dtypes[0],
                'crs': str(src.crs),
                'transform': src.transform,
                'bounds': src.bounds
            }
            return info
    except Exception as e:
        # Попробуем обработать ошибку кодировки
        try:
            # Просто проверим, существует ли файл
            import pathlib
            path = pathlib.Path(orthophoto_path)
            if path.exists():
                print(f"Файл существует, но возникли проблемы с чтением информации: {str(e)}")
                return {
                    'width': 'unknown',
                    'height': 'unknown',
                    'count': 'unknown',
                    'dtype': 'unknown',
                    'crs': 'unknown',
                    'transform': 'unknown',
                    'bounds': 'unknown'
                }
        except Exception:
            pass
        raise Exception(f"Ошибка получения информации об ортофотоплане: {str(e)}")