import numpy as np
from PIL import Image
from PIL import ImageOps
import math


class Renderer:
    def __init__(self, width=1024, height=1024, projection_factor=0.2, projection_distance=5000, offset_x=500,
                 offset_y=500):
        """
        Инициализация рендера

        :param width: Ширина изображения в пикселях
        :param height: Высота изображения в пикселях
        :param projection_factor: Коэффициент масштаба проекции
        :param projection_distance: Дистанция проекции
        :param offset_x: Смещение по X для центрирования изображения
        :param offset_y: Смещение по Y для центрирования изображения
        """
        self.width = width
        self.height = height
        self.projection_factor = projection_factor
        self.projection_distance = projection_distance
        self.offset_x = offset_x
        self.offset_y = offset_y

        # Инициализация буферов изображения и глубины
        self.img = np.zeros((height, width, 3), dtype=np.uint8)
        self.z_buf = np.matrix(np.inf * np.ones((height, width)))

    def clear_buffers(self):
        """Очистка буферов изображения и глубины"""
        self.img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.z_buf = np.matrix(np.inf * np.ones((self.height, self.width)))

    def save_image(self, target_path):
        """
        Сохранение изображения в файл

        :param target_path: Путь для сохранения файла изображения
        """
        # Корректируем ориентацию изображения перед сохранением
        image = Image.fromarray(np.flip(self.img, 1))
        image = ImageOps.flip(image)
        image.save(target_path)

    def render_model(self, model):
        """
        Рендеринг 3D-модели на изображение

        :param model: Объект модели для рендеринга
        """
        # Вычисляем нормали для вершин модели
        self._calculate_vertex_normals(model)

        # Отрисовываем каждый полигон модели
        for i, face in enumerate(model.faces):
            # Отрисовка треугольника
            if len(face) == 3:
                self._render_triangle(model, face, i)
            # Разбиваем полигон на треугольники, если в нём более 3 вершин
            else:
                for j in range(1, len(face) - 1):
                    triangle_face = [face[0], face[j], face[j + 1]]
                    self._render_triangle(model, triangle_face, i)

    def _calculate_vertex_normals(self, model):
        """
        Вычисление нормалей для вершин модели

        :param model: Объект модели
        """
        # Инициализируем массив нормалей для вершин
        model.vertex_normals = np.zeros((len(model.vertices), 3))

        # Вычисляем нормали для каждого полигона и добавляем их к соответствующим вершинам
        for face in model.faces:
            if len(face) >= 3:  # Проверяем, что полигон имеет хотя бы 3 вершины
                # Берем первые три вершины для вычисления нормали
                v1 = model.vertices[face[0]][:3]
                v2 = model.vertices[face[1]][:3]
                v3 = model.vertices[face[2]][:3]

                # Вычисляем нормаль к полигону
                nx, ny, nz = self._calculate_normal(v1[0], v1[1], v1[2], v2[0], v2[1], v2[2], v3[0], v3[1], v3[2])

                # Добавляем нормаль к каждой вершине полигона
                for vertex_idx in face:
                    model.vertex_normals[vertex_idx] += np.array([nx, ny, nz])

        # Нормализуем нормали
        for i in range(len(model.vertex_normals)):
            norm = np.linalg.norm(model.vertex_normals[i])
            if norm > 0:
                model.vertex_normals[i] = model.vertex_normals[i] / norm

    def _calculate_normal(self, x0, y0, z0, x1, y1, z1, x2, y2, z2):
        """
        Вычисление нормали к треугольнику

        :return: Компоненты нормали (x, y, z)
        """
        # Векторное произведение для вычисления нормали
        x = (y1 - y2) * (z1 - z0) - (z1 - z2) * (y1 - y0)
        y = (x1 - x2) * (z1 - z0) - (z1 - z2) * (x1 - x0)
        z = (x1 - x2) * (y1 - y0) - (y1 - y2) * (x1 - x0)
        return (x, y, z)

    def _calculate_barycentric(self, x, y, x0, y0, x1, y1, x2, y2):
        """
        Вычисление барицентрических координат точки относительно треугольника

        :return: Барицентрические координаты (lambda0, lambda1, lambda2)
        """
        denominator = (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)
        if abs(denominator) > 1e-6:
            lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / denominator
            lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / denominator
            lambda2 = 1.0 - lambda0 - lambda1
            return (lambda0, lambda1, lambda2)
        else:
            raise Exception("Не треугольник или треугольник вырожденный")

    def _render_triangle(self, model, face, face_index):
        """
        Отрисовка треугольника с учетом z-буфера и текстур

        :param model: Объект модели
        :param face: Индексы вершин треугольника
        :param face_index: Индекс полигона
        """
        if len(face) != 3:
            return  # Убедимся, что это треугольник

        # Получаем координаты вершин треугольника
        v1 = model.vertices[face[0]]
        v2 = model.vertices[face[1]]
        v3 = model.vertices[face[2]]

        # Извлекаем координаты
        x0, y0, z0 = v1[0], v1[1], v1[2]
        x1, y1, z1 = v2[0], v2[1], v2[2]
        x2, y2, z2 = v3[0], v3[1], v3[2]

        # Проверяем, что точки не находятся за камерой
        if z0 <= 0 or z1 <= 0 or z2 <= 0:
            return

        # Проецируем 3D-координаты на 2D-плоскость
        px0 = self.projection_factor * self.projection_distance * x0 / z0 + self.offset_x
        px1 = self.projection_factor * self.projection_distance * x1 / z1 + self.offset_x
        px2 = self.projection_factor * self.projection_distance * x2 / z2 + self.offset_x
        py0 = self.projection_factor * self.projection_distance * y0 / z0 + self.offset_y
        py1 = self.projection_factor * self.projection_distance * y1 / z1 + self.offset_y
        py2 = self.projection_factor * self.projection_distance * y2 / z2 + self.offset_y

        # Определяем границы треугольника с учетом границ изображения
        x_min = max(0, min(px0, px1, px2))
        y_min = max(0, min(py0, py1, py2))
        x_max = min(self.width - 1, max(px0, px1, px2))
        y_max = min(self.height - 1, max(py0, py1, py2))

        # Вычисляем нормаль треугольника для определения лицевой стороны
        nx, ny, nz = self._calculate_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
        cos = nz / (math.sqrt(nx ** 2 + ny ** 2 + nz ** 2) + 1e-10)

        # Отрисовываем только лицевые полигоны (направленные к камере)
        if cos < 0:
            # Получаем нормали к вершинам для тонировки Гуро
            n0 = model.vertex_normals[face[0]][2]  # Z-компонента нормали
            n1 = model.vertex_normals[face[1]][2]
            n2 = model.vertex_normals[face[2]][2]

            # Получаем текстурные координаты, если они есть
            if model.has_texture and face_index < len(model.texture_faces):
                tex_face = model.texture_faces[face_index]
                if len(tex_face) >= 3:  # Убедимся, что у нас есть текстурные координаты
                    t1 = model.texture_vertices[tex_face[0]]
                    t2 = model.texture_vertices[tex_face[1]]
                    t3 = model.texture_vertices[tex_face[2]]
                    has_texture_coords = True
                else:
                    has_texture_coords = False
            else:
                has_texture_coords = False

            # Растеризация треугольника
            for x in range(int(x_min), int(x_max) + 1):
                for y in range(int(y_min), int(y_max) + 1):
                    try:
                        # Вычисляем барицентрические координаты
                        lambda0, lambda1, lambda2 = self._calculate_barycentric(x, y, px0, py0, px1, py1, px2, py2)

                        # Проверяем, находится ли точка внутри треугольника
                        if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0:
                            # Интерполируем z-значение
                            z = lambda0 * z0 + lambda1 * z1 + lambda2 * z2

                            # Проверяем z-буфер
                            if z < self.z_buf[y, x]:
                                self.z_buf[y, x] = z

                                # Интерполируем нормаль для тонировки Гуро
                                intensity = -(lambda0 * n0 + lambda1 * n1 + lambda2 * n2)
                                intensity = max(0.1, min(1.0, intensity))  # Ограничиваем значение

                                # Применяем текстуру, если она доступна
                                if model.has_texture and has_texture_coords and model.texture is not None:
                                    # Интерполируем текстурные координаты
                                    tex_u = lambda0 * t1[0] + lambda1 * t2[0] + lambda2 * t3[0]
                                    tex_v = lambda0 * t1[1] + lambda1 * t2[1] + lambda2 * t3[1]

                                    # Получаем цвет из текстуры
                                    tex_x = int(model.texture.shape[1] * tex_u) % model.texture.shape[1]
                                    tex_y = int(model.texture.shape[0] * tex_v) % model.texture.shape[0]
                                    color = model.texture[tex_y, tex_x] * intensity
                                else:
                                    # Если текстуры нет, используем простой серый цвет
                                    color = np.array([200, 200, 200]) * intensity

                                self.img[y, x] = np.clip(color, 0, 255).astype(np.uint8)
                    except Exception:
                        continue  # Пропускаем точки, для которых невозможно вычислить барицентрические координаты


class Model:
    def __init__(self):
        """Инициализация модели"""
        self.vertices = []  # Вершины модели
        self.texture_vertices = []  # Текстурные координаты
        self.faces = []  # Полигоны (индексы вершин)
        self.texture_faces = []  # Полигоны текстуры (индексы текстурных координат)
        self.vertex_normals = []  # Нормали вершин
        self.has_texture = False  # Флаг наличия текстуры
        self.texture = None  # Изображение текстуры

    def load_from_obj(self, filename):
        """
        Загрузка модели из OBJ-файла

        :param filename: Путь к OBJ-файлу
        """
        # Очищаем данные модели перед загрузкой
        self.vertices = []
        self.texture_vertices = []
        self.faces = []
        self.texture_faces = []

        vertex_index = 0
        texture_vertex_index = 0

        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()

                # Пропускаем пустые строки
                if not line:
                    continue

                parts = line.split()

                if len(parts) == 0:
                    continue

                # Обработка вершин
                if parts[0] == 'v':
                    vertex = [float(coord) for coord in parts[1:4]]  # Берем только x, y, z
                    vertex.append(vertex_index)  # Добавляем индекс вершины
                    vertex_index += 1
                    self.vertices.append(vertex)

                # Обработка текстурных координат
                elif parts[0] == 'vt':
                    tex_vertex = [float(coord) for coord in parts[1:3]]  # Берем только u, v
                    tex_vertex.append(texture_vertex_index)  # Добавляем индекс текстурной координаты
                    texture_vertex_index += 1
                    self.texture_vertices.append(tex_vertex)
                    self.has_texture = True

                # Обработка полигонов
                elif parts[0] == 'f':
                    # Извлекаем индексы вершин и текстурных координат
                    face_vertices = []
                    face_textures = []

                    for part in parts[1:]:
                        indices = part.split('/')

                        # Индекс вершины (с учетом того, что в OBJ нумерация начинается с 1)
                        if len(indices) > 0 and indices[0]:
                            face_vertices.append(int(indices[0]) - 1)

                        # Индекс текстурной координаты
                        if len(indices) > 1 and indices[1]:
                            face_textures.append(int(indices[1]) - 1)

                    self.faces.append(face_vertices)

                    if len(face_textures) > 0:
                        self.texture_faces.append(face_textures)

        # Конвертируем списки в массивы numpy для эффективности
        self.vertices = np.array(self.vertices)
        if self.texture_vertices:
            self.texture_vertices = np.array(self.texture_vertices)

    def load_texture(self, texture_path):
        """
        Загрузка текстуры из файла изображения

        :param texture_path: Путь к файлу текстуры
        """
        if texture_path:
            self.texture = np.asarray(Image.open(texture_path))
            # Инвертируем текстуру по Y для соответствия координатам OBJ
            self.texture = self.texture[::-1]
            self.has_texture = True

    def apply_transformation(self, translation=(0, 0, 0), rotation_euler=(0, 0, 0), scale=1.0, use_quaternion=False,
                             quaternion=None):
        """
        Применение 3D-трансформаций к модели

        :param translation: Смещение (tx, ty, tz)
        :param rotation_euler: Углы поворота Эйлера в градусах (rx, ry, rz)
        :param scale: Коэффициент масштабирования
        :param use_quaternion: Использовать кватернионы вместо углов Эйлера
        :param quaternion: Кватернион поворота (w, x, y, z)
        """
        # Копируем вершины перед трансформацией
        transformed_vertices = self.vertices.copy()

        # Применяем масштаб
        transformed_vertices[:, :3] *= scale

        # Применяем поворот
        if use_quaternion and quaternion is not None:
            # Применяем поворот с использованием кватерниона
            rotation_matrix = self._quaternion_to_rotation_matrix(quaternion)
            transformed_vertices[:, :3] = transformed_vertices[:, :3] @ rotation_matrix.T
        else:
            # Применяем поворот с использованием углов Эйлера
            rotation_matrix = self._euler_to_rotation_matrix(rotation_euler)
            transformed_vertices[:, :3] = transformed_vertices[:, :3] @ rotation_matrix.T

        # Применяем смещение
        transformed_vertices[:, :3] += translation

        # Обновляем вершины модели
        self.vertices = transformed_vertices

    def _euler_to_rotation_matrix(self, euler_angles):
        """
        Преобразование углов Эйлера в матрицу поворота

        :param euler_angles: Углы Эйлера в градусах (rx, ry, rz)
        :return: Матрица поворота 3x3
        """
        rx, ry, rz = [np.radians(angle) for angle in euler_angles]

        # Матрица поворота вокруг оси X
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(rx), np.sin(rx)],
            [0, -np.sin(rx), np.cos(rx)]
        ])

        # Матрица поворота вокруг оси Y
        R_y = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])

        # Матрица поворота вокруг оси Z
        R_z = np.array([
            [np.cos(rz), np.sin(rz), 0],
            [-np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])

        # Комбинируем матрицы поворота (порядок: Z * Y * X)
        R = R_x @ R_y @ R_z
        return R

    def _quaternion_to_rotation_matrix(self, quaternion):
        """
        Преобразование кватерниона в матрицу поворота

        :param quaternion: Кватернион (w, x, y, z)
        :return: Матрица поворота 3x3
        """
        w, x, y, z = quaternion

        # Нормализуем кватернион
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        if norm > 0:
            w, x, y, z = w / norm, x / norm, y / norm, z / norm

        # Формируем матрицу поворота
        xx, xy, xz = x * x, x * y, x * z
        yy, yz, zz = y * y, y * z, z * z
        wx, wy, wz = w * x, w * y, w * z

        R = np.array([
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
        ])

        return R


def render_with_gourad_and_texture(model_path, texture_path, target_path,
                                   width=1024, height=1024,
                                   translation=(0, -0.04, 0.2),
                                   rotation_euler=(0, 120, 17),
                                   scale=1.0):
    """
    Рендеринг модели с текстурой и тонировкой Гуро

    :param model_path: Путь к OBJ-файлу модели
    :param texture_path: Путь к файлу текстуры
    :param target_path: Путь для сохранения результата
    :param width: Ширина изображения
    :param height: Высота изображения
    :param translation: Смещение модели (tx, ty, tz)
    :param rotation_euler: Углы поворота в градусах (rx, ry, rz)
    :param scale: Коэффициент масштабирования
    """
    # Создаем рендерер
    renderer = Renderer(width=width, height=height)

    # Загружаем модель из OBJ-файла
    model = Model()
    model.load_from_obj(model_path)

    # Загружаем текстуру, если путь указан
    if texture_path:
        model.load_texture(texture_path)

    # Применяем трансформации к модели
    model.apply_transformation(translation=translation, rotation_euler=rotation_euler, scale=scale)

    # Рендерим модель
    renderer.render_model(model)

    # Сохраняем результат
    renderer.save_image(target_path)


def render_multiple_models(models_data, target_path, width=1024, height=1024):
    """
    Рендеринг нескольких моделей на одном изображении

    :param models_data: Список словарей с параметрами моделей
                       [{
                           'model_path': путь к OBJ-файлу,
                           'texture_path': путь к текстуре,
                           'translation': (tx, ty, tz),
                           'rotation_euler': (rx, ry, rz),
                           'scale': масштаб
                       }, ...]
    :param target_path: Путь для сохранения результата
    :param width: Ширина изображения
    :param height: Высота изображения
    """
    # Создаем рендерер
    renderer = Renderer(width=width, height=height, offset_x = 500, offset_y = 500)

    # Рендерим каждую модель по очереди
    for model_data in models_data:
        model = Model()
        model.load_from_obj(model_data['model_path'])

        if 'texture_path' in model_data and model_data['texture_path']:
            model.load_texture(model_data['texture_path'])

        # Применяем трансформации
        translation = model_data.get('translation', (0, 0, 0))
        rotation_euler = model_data.get('rotation_euler', (0, 0, 0))
        rotation_quaternion = model_data.get('rotation_quaternion', None)
        scale = model_data.get('scale', 1.0)

        # Проверяем, какой тип поворота использовать
        use_quaternion = 'rotation_quaternion' in model_data and rotation_quaternion is not None

        model.apply_transformation(
            translation=translation,
            rotation_euler=rotation_euler,
            scale=scale,
            use_quaternion=use_quaternion,
            quaternion=rotation_quaternion
        )

        # Рендерим модель (z-буфер сохраняется между моделями)
        renderer.render_model(model)

    # Сохраняем итоговое изображение
    renderer.save_image(target_path)


if __name__ == "__main__":

    # Рендеринг одной модели
    render_with_gourad_and_texture(
        '../LR_4/12221_Cat_v1_l3.obj',
        '../LR_4/Cat_diffuse.jpg',
        'textured_1.png',
        width=1024,
        height=1024,
        translation = (0, -6, 20),
        rotation_euler=(90, 0, 180),
        scale=0.3
    )

    '''
    # рендеринг нескольких моделей
    models_data = [
        {
            'model_path': '../model.obj',
            'texture_path': 'bunny-atlas.jpg',
            'translation': (0, -0.04, 0.2),
            'rotation_euler': (0, 120, 17),
            'scale': 1.0
        },
        {
            'model_path': '../model.obj',
            'texture_path': 'bunny-atlas.jpg',
            'translation': (0.062, -0.01, 0.225),
            'rotation_euler': (0, -50, 0),
            'scale': 0.7
        }
    ]

    render_multiple_models(models_data, 'multiple_models.png')'''
    shizocat_rotation = (90, 0, 180)
    models_data = [
        {
            'model_path': '../LR_4/12221_Cat_v1_l3.obj',
            'texture_path': '../LR_4/Cat_diffuse.jpg',
            'translation': (0, -6, 20),
            'rotation_euler': shizocat_rotation,
            'scale': 0.3
        },
        {
            'model_path': '../LR_4/12221_Cat_v1_l3.obj',
            'texture_path': '../LR_4/Cat_diffuse.jpg',
            'translation': (-2, -6, 20),
            'rotation_euler': shizocat_rotation,
            'scale': 0.3
        },
        {
            'model_path': '../LR_4/12221_Cat_v1_l3.obj',
            'texture_path': '../LR_4/Cat_diffuse.jpg',
            'translation': (2, -6, 20),
            'rotation_euler': shizocat_rotation,
            'scale': 0.3
        },
        {
            'model_path': '../LR_4/12221_Cat_v1_l3.obj',
            'texture_path': '../LR_4/Cat_diffuse.jpg',
            'translation': (-4, -6, 20),
            'rotation_euler': shizocat_rotation,
            'scale': 0.3
        },
        {
            'model_path': '../LR_4/12221_Cat_v1_l3.obj',
            'texture_path': '../LR_4/Cat_diffuse.jpg',
            'translation': (4, -6, 20),
            'rotation_euler': shizocat_rotation,
            'scale': 0.3
        },
        {
            'model_path': '../LR_4/12221_Cat_v1_l3.obj',
            'texture_path': '../LR_4/Cat_diffuse.jpg',
            'translation': (-6, -6, 20),
            'rotation_euler': shizocat_rotation,
            'scale': 0.3
        },
        {
            'model_path': '../LR_4/12221_Cat_v1_l3.obj',
            'texture_path': '../LR_4/Cat_diffuse.jpg',
            'translation': (6, -6, 20),
            'rotation_euler': shizocat_rotation,
            'scale': 0.3
        }
    ]

    render_multiple_models(models_data, 'shizocats.png')
