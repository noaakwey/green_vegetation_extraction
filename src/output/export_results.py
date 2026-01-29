"""
Модуль экспорта результатов выделения зеленых насаждений
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point
from skimage.measure import regionprops
import json
from typing import List, Dict
import os

def export_to_geojson(regions: List,
                     output_path: str,
                     crs: str = "EPSG:4326") -> None:
    """
    Экспорт результатов в формат GeoJSON

    Args:
        regions (List): Список регионов (объектов)
        output_path (str): Путь для сохранения файла
        crs (str): Проекция координат
    """
    # Создание списка геометрий
    geometries = []
    properties = []

    for region in regions:
        # Создание полигона из координат региона
        coords = region.coords
        polygon = Polygon(coords)

        geometries.append(polygon)

        # Добавление свойств объекта
        properties.append({
            'area': region.area,
            'perimeter': region.perimeter,
            'centroid_x': region.centroid[0],
            'centroid_y': region.centroid[1],
            'circularity': 4 * np.pi * region.area / (region.perimeter ** 2) if region.perimeter > 0 else 0
        })

    # Создание GeoDataFrame
    gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs=crs)

    # Сохранение в файл
    gdf.to_file(output_path, driver='GeoJSON')

def export_to_shapefile(regions: List,
                       output_path: str,
                       crs: str = "EPSG:4326") -> None:
    """
    Экспорт результатов в формат Shapefile

    Args:
        regions (List): Список регионов (объектов)
        output_path (str): Путь для сохранения файла (без расширения)
        crs (str): Проекция координат
    """
    # Создание списка геометрий
    geometries = []
    properties = []

    for region in regions:
        # Создание полигона из координат региона
        coords = region.coords
        polygon = Polygon(coords)

        geometries.append(polygon)

        # Добавление свойств объекта
        properties.append({
            'area': region.area,
            'perimeter': region.perimeter,
            'centroid_x': region.centroid[0],
            'centroid_y': region.centroid[1],
            'circularity': 4 * np.pi * region.area / (region.perimeter ** 2) if region.perimeter > 0 else 0
        })

    # Создание GeoDataFrame
    gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs=crs)

    # Сохранение в файл
    gdf.to_file(output_path + '.shp', driver='ESRI Shapefile')

def export_to_csv(regions: List,
                 output_path: str) -> None:
    """
    Экспорт результатов в формат CSV

    Args:
        regions (List): Список регионов (объектов)
        output_path (str): Путь для сохранения файла
    """
    # Создание списка данных
    data = []

    for region in regions:
        data.append({
            'area': region.area,
            'perimeter': region.perimeter,
            'centroid_x': region.centroid[0],
            'centroid_y': region.centroid[1],
            'circularity': 4 * np.pi * region.area / (region.perimeter ** 2) if region.perimeter > 0 else 0
        })

    # Сохранение в CSV
    import pandas as pd
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

def export_results_summary(regions: List,
                          output_path: str) -> None:
    """
    Экспорт сводной информации о результатах

    Args:
        regions (List): Список регионов (объектов)
        output_path (str): Путь для сохранения файла
    """
    summary = {
        'total_objects': len(regions),
        'total_area': sum([region.area for region in regions]),
        'average_area': np.mean([region.area for region in regions]) if regions else 0,
        'min_area': min([region.area for region in regions]) if regions else 0,
        'max_area': max([region.area for region in regions]) if regions else 0,
        'average_circularity': np.mean([4 * np.pi * region.area / (region.perimeter ** 2)
                                       if region.perimeter > 0 else 0
                                       for region in regions]) if regions else 0
    }

    # Сохранение в JSON
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

def export_to_image(binary_mask: np.ndarray,
                   output_path: str) -> None:
    """
    Экспорт бинарной маски в виде изображения

    Args:
        binary_mask (np.ndarray): Бинарная маска
        output_path (str): Путь для сохранения изображения
    """
    from PIL import Image

    # Преобразование маски в изображение
    img = Image.fromarray(binary_mask.astype(np.uint8))
    img.save(output_path)