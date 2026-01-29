"""
Тесты для модулей классификации
"""

import unittest
import numpy as np
from src.classification.threshold_classification import threshold_classification, combined_classification
from src.data_processing.orthophoto_loader import load_orthophoto
from src.spectral_analysis.spectral_indices import calculate_color_indices

class TestClassification(unittest.TestCase):
    """Тесты для модулей классификации"""

    def test_color_indices_calculation(self):
        """Тест вычисления цветовых индексов"""
        # Создаем тестовое RGB изображение
        test_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Вычисляем индексы
        indices = calculate_color_indices(test_data)

        # Проверяем, что все индексы вычислены
        self.assertIn('green_red_ratio', indices)
        self.assertIn('green_blue_ratio', indices)
        self.assertIn('green_index', indices)

        # Проверяем размеры
        self.assertEqual(indices['green_red_ratio'].shape, (100, 100))
        self.assertEqual(indices['green_blue_ratio'].shape, (100, 100))
        self.assertEqual(indices['green_index'].shape, (100, 100))

    def test_threshold_classification(self):
        """Тест пороговой классификации"""
        # Создаем тестовое RGB изображение с явно зелеными участками
        test_data = np.zeros((100, 100, 3), dtype=np.uint8)

        # Добавляем зеленый участок
        test_data[20:80, 20:80, 1] = 200  # зеленый канал
        test_data[20:80, 20:80, 0] = 50   # красный канал
        test_data[20:80, 20:80, 2] = 50   # синий канал

        # Применяем классификацию
        mask = threshold_classification(test_data)

        # Проверяем, что маска создана
        self.assertEqual(mask.shape, (100, 100))
        self.assertIn(0, np.unique(mask))
        self.assertIn(255, np.unique(mask))

    def test_combined_classification(self):
        """Тест комбинированной классификации"""
        # Создаем тестовое RGB изображение
        test_data = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

        # Применяем комбинированную классификацию
        mask = combined_classification(test_data)

        # Проверяем, что маска создана
        self.assertEqual(mask.shape, (50, 50))
        self.assertIn(0, np.unique(mask))
        self.assertIn(255, np.unique(mask))

if __name__ == '__main__':
    unittest.main()