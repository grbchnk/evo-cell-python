import pygame
import pygame_gui
import sys
import random
import numpy as np
import time
from collections import deque

import numpy as np

import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model

WORLD_WIDTH = 960
WORLD_HEIGHT = 640
WORLD_SCALE_FACTOR = 20

class FPS:
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()

    def count(self):
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= 1:  # каждую секунду
            print(f"FPS: {self.frame_count}")
            self.frame_count = 0
            self.start_time = time.time()

class Perceptron:
    def __init__(self, input_dim, hidden_dim, output_dim):
        input_layer = Input(shape=(input_dim,))
        hidden_layer_1 = Dense(hidden_dim, activation='relu')(input_layer)
        # hidden_layer_2 = Dense(hidden_dim, activation='relu')(hidden_layer_1)
        output_layer = Dense(output_dim, activation='softmax')(hidden_layer_1)
        
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def get_weights(self):
        weights = []
        for layer in self.model.layers:
            weights.append(layer.get_weights())
        return weights

    def set_weights(self, weights):
        for i, layer in enumerate(self.model.layers):
            layer.set_weights(weights[i])

    def print_weights(self):
        weights = self.get_weights()
        for i, layer_weights in enumerate(weights):
            if len(layer_weights) > 0:  # Проверяем, есть ли веса
                print(f"weights layer {i + 1}:")
                print("weights:", layer_weights[0])
                print("offsets:", layer_weights[1])
                print()

    def predict(self, x):
        x = np.array(x).reshape(1, -1)  # Преобразование в двумерный массив
        return self.model.predict(x)

class Snake:
    DIRECTION_UP = 0
    DIRECTION_RIGHT = 1
    DIRECTION_DOWN = 2
    DIRECTION_LEFT = 3
    VIEW_RADIUS = 10
    MAX_ENERGY = 100
    MAX_LENGTH = 3

    MUTATION_RATE = 0.1  # вероятность мутации каждого веса
    MUTATION_SCALE = 1.0  # масштаб (стандартное отклонение) мутаций

    def __init__(self, position, direction, body, energy=MAX_ENERGY):
        self.position = position
        self.direction = direction
        self.body = deque(body)  # Используйте deque вместо списка
        # self.body = body  # Используйте deque вместо списка
        self.body_set = set(body)  # Создайте хеш-сет для тела змейки
        self.energy = energy
        self.brain = Perceptron(6, 2, 3)  # Добавляем мозг в виде перцептрона

    def step(self):
        # self.energy -= 1  # змейка тратит энергию на движение
        
        if self.energy <= 0:
            self.energy = self.MAX_ENERGY
            self.body_set.remove(self.body.pop())  # если энергия заканчивается, змейка уменьшает свое тело на 1

        if len(self.body) > 0:
            closest_objects, closest_distances = self.look_around(self.VIEW_RADIUS)

            # Преобразование данных в подходящий формат
            input_data = closest_objects + closest_distances

            # Получение предсказания от перцептрона
            # prediction = self.brain.predict(input_data)

            # Выбор действия с наибольшей вероятностью
            actions = random.choice(["FORWARD", "LEFT", "RIGHT"])
            # action = actions[np.argmax(prediction)]

            snake.move(actions)
        else:
            self.dead()

    def move(self, direction):
        if len(self.body) >= self.MAX_LENGTH:
            self.reproduce()
        # self.brain.print_weights()
        new_direction = self.change_direction(direction)

        if new_direction == self.DIRECTION_UP:
            next_position = (self.position[0], (self.position[1] - WORLD_SCALE_FACTOR) % WORLD_HEIGHT)
        elif new_direction == self.DIRECTION_RIGHT:
            next_position = ((self.position[0] + WORLD_SCALE_FACTOR) % WORLD_WIDTH, self.position[1])
        elif new_direction == self.DIRECTION_DOWN:
            next_position = (self.position[0], (self.position[1] + WORLD_SCALE_FACTOR) % WORLD_HEIGHT)
        elif new_direction == self.DIRECTION_LEFT:
            next_position = ((self.position[0] - WORLD_SCALE_FACTOR) % WORLD_WIDTH, self.position[1])
        else:
            return    
        
        for snake in field.snakes:
            if snake != self and next_position in snake.body_set:
                self.dead()
                return
        
        if next_position in self.body_set:
            # то удаляем из тела все элементы, начиная с этой позиции.
            while self.body[-1] != next_position:
                self.body_set.remove(self.body.pop())
            self.body_set.remove(self.body.pop())
        elif next_position in field.wall_set:
            self.dead()
        elif next_position in field.food_set:
            self.eat()
            field.foods.remove(next_position)
            field.add_food(field.random_food())

            self.position = next_position
            self.direction = new_direction

            self.body.appendleft(self.position)
            self.body_set.add(self.position)
        else:
            # Если следующая позиция свободна,
            self.position = next_position
            self.direction = new_direction

            # Добавляем новую позицию в начало тела змейки.
            self.body.appendleft(self.position)
            self.body_set.add(self.position)

            if len(self.body) > 1:
                self.body_set.remove(self.body.pop())

    def get_direction(self, pos1, pos2):
        # Вычисляем направление от pos1 к pos2
        dx, dy = pos2[0] - pos1[0], pos2[1] - pos1[1]
        if dx > 0:
            return self.DIRECTION_RIGHT
        elif dx < 0:
            return self.DIRECTION_LEFT
        elif dy > 0:
            return self.DIRECTION_DOWN
        else:
            return self.DIRECTION_UP

    def change_direction(self, direction):
        if direction == "FORWARD":
            new_direction = self.direction
        elif direction == "LEFT":
            new_direction = (self.direction - 1) % 4  # Поворот налево
        elif direction == "RIGHT":
            new_direction = (self.direction + 1) % 4  # Поворот направо

        return new_direction
        
    def look_around(self, view_radius):
        # Определение направлений взгляда
        directions = [(self.direction - 1) % 4, self.direction, (self.direction + 1) % 4]
        closest_objects = [0, 0, 0]
        closest_distances = [view_radius + 1, view_radius + 1, view_radius + 1]

        # Проверка каждой позиции в радиусе обзора
        for distance in range(1, view_radius + 1):
            for direction in directions:
                if direction == self.DIRECTION_UP:
                    pos = (self.position[0], (self.position[1] - distance * WORLD_SCALE_FACTOR) % WORLD_HEIGHT)
                elif direction == self.DIRECTION_RIGHT:
                    pos = ((self.position[0] + distance * WORLD_SCALE_FACTOR) % WORLD_WIDTH, self.position[1])
                elif direction == self.DIRECTION_DOWN:
                    pos = (self.position[0], (self.position[1] + distance * WORLD_SCALE_FACTOR) % WORLD_HEIGHT)
                elif direction == self.DIRECTION_LEFT:
                    pos = ((self.position[0] - distance * WORLD_SCALE_FACTOR) % WORLD_WIDTH, self.position[1])

                # Проверка столкновений со стенами, едой и другими змейками
                if pos in field.wall_set:
                    object_type = 3
                elif pos in field.snake_set:
                    object_type = 2
                elif pos in field.food_set:
                    object_type = 1
                else:
                    continue

                # Обновление ближайших объектов и расстояний
                for k in range(3):
                    if directions[k] == direction and distance < closest_distances[k]:
                        closest_objects[k] = object_type
                        closest_distances[k] = round(1 - distance / view_radius, 2)

        for k in range(3):
            if closest_objects[k] == 0:
                closest_distances[k] = 0

        return closest_objects, closest_distances

    def reproduce(self):
        half_length = len(self.body) // 2

        # Вычисляем направление хвоста старой змейки
        tail_direction = self.get_direction(self.body[-2], self.body[-1])

        # Создаем новую змейку с половиной тела старой змейки
        body_list = list(self.body)  # Преобразуем deque в список для использования среза
        new_body = body_list[half_length:]
        new_body.reverse()  # Разворачиваем список тела второй змейки

        child = Snake(new_body[0], tail_direction, new_body)

        # Удаляем половину тела у старой змейки
        self.body = deque(body_list[:half_length])
        self.body_set = set(self.body)

        # Копирование весов
        child.brain.set_weights(self.brain.get_weights())

        weights = child.brain.get_weights()
        for i in range(len(weights)):
            if len(weights[i]) > 0:  # Проверяем, есть ли веса
                mutation_mask = np.random.uniform(0., 1., size=weights[i][0].shape) < self.MUTATION_RATE
                mutation_values = np.random.standard_normal(size=weights[i][0].shape) * self.MUTATION_SCALE
                weights[i][0] += mutation_mask * mutation_values
        child.brain.set_weights(weights)

        # Добавляем новую змейку на поле
        field.add_snake(child)

    def eat(self):
        return

    def dead(self):
        field.snakes.remove(self)

class Field:
    def __init__(self, size):
        self.size = size
        self.snakes = []
        self.foods = []
        self.walls = []
        self.update_sets()

    def update_sets(self):
        self.wall_set = set(self.walls)
        self.food_set = set(self.foods)
        self.snake_set = set(pos for snake in self.snakes for pos in snake.body)

    def add_wall(self, wall):
        self.walls.append(wall)
        self.update_sets()

    def add_food(self, food):
        self.foods.append(food)
        self.update_sets()

    def add_snake(self, snake):
        self.snakes.append(snake)
        self.update_sets()

    def random_food(self):
        while True:
            position = (random.randint(0, self.size[0]//WORLD_SCALE_FACTOR - 1)*WORLD_SCALE_FACTOR, random.randint(0, self.size[1]//WORLD_SCALE_FACTOR - 1)*WORLD_SCALE_FACTOR)
            if position not in [segment for snake in self.snakes for segment in snake.body] and position not in self.walls:
                return position

    def random_wall(self):
        while True:
            position = (random.randint(0, self.size[0]//WORLD_SCALE_FACTOR - 1)*WORLD_SCALE_FACTOR, random.randint(0, self.size[1]//WORLD_SCALE_FACTOR - 1)*WORLD_SCALE_FACTOR)
            if position not in [segment for snake in self.snakes for segment in snake.body] and position not in self.foods:
                return position

    def random_snake(self):
        while True:
            position = (random.randint(0, self.size[0]//WORLD_SCALE_FACTOR - 1)*WORLD_SCALE_FACTOR, random.randint(0, self.size[1]//WORLD_SCALE_FACTOR - 1)*WORLD_SCALE_FACTOR)
            if position not in self.walls and position not in self.foods:
                return Snake(position, random.choice([Snake.DIRECTION_UP, Snake.DIRECTION_RIGHT, Snake.DIRECTION_DOWN, Snake.DIRECTION_LEFT]), [position])

pygame.init()


screen = pygame.display.set_mode((WORLD_WIDTH, WORLD_HEIGHT))
field = Field((WORLD_WIDTH, WORLD_HEIGHT))
fps_counter = FPS()

for _ in range(50):
    field.add_food(field.random_food())

for _ in range(50):
    field.add_wall(field.random_wall())

for _ in range(100):
    field.add_snake(field.random_snake())

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            field.update_sets()
            if event.key == pygame.K_UP:
                for snake in field.snakes:
                    snake.move("FORWARD")
            elif event.key == pygame.K_RIGHT:
                for snake in field.snakes:
                    snake.move("RIGHT")
            elif event.key == pygame.K_LEFT:
                for snake in field.snakes:
                    snake.move("LEFT")
            elif event.key == pygame.K_SPACE:
                for snake in field.snakes:
                    snake.step()

    screen.fill((0, 0, 0))

    # fps_counter.count()

    for snake in field.snakes:
        snake.step()

    for snake in field.snakes:
        for segment in snake.body:
            pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(segment[0], segment[1], WORLD_SCALE_FACTOR, WORLD_SCALE_FACTOR))

    for food in field.foods:
        pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(food[0], food[1], WORLD_SCALE_FACTOR, WORLD_SCALE_FACTOR))

    for wall in field.walls:
        pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(wall[0], wall[1], WORLD_SCALE_FACTOR, WORLD_SCALE_FACTOR))

    # Обновление экрана
    pygame.display.flip()

    # Задержка
    # pygame.time.delay(50)
