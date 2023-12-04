import pygame
import sys
import random
import numpy as np
import time

WORLD_WIDTH = 960
WORLD_HEIGHT = 640
WORLD_SCALE_FACTOR = 40

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

class Snake:
    DIRECTION_UP = 0
    DIRECTION_RIGHT = 1
    DIRECTION_DOWN = 2
    DIRECTION_LEFT = 3
    VIEW_RADIUS = 10
    MAX_ENERGY = 100

    def __init__(self, position, direction, body, energy=MAX_ENERGY):
        self.position = position
        self.direction = direction
        self.body = body
        self.energy = energy

    def step(self):
        # self.energy -= 1  # змейка тратит энергию на движение

        if self.energy <= 0:
            self.energy = self.MAX_ENERGY
            self.body.pop()  # если энергия заканчивается, змейка уменьшает свое тело на 1

        if len(self.body) > 0:
            self.look_around(self.VIEW_RADIUS)

            snake.move(random.choice(["FORWARD", "LEFT", "RIGHT"]))
            return
        else:
            self.dead()

    def move(self, direction):
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
        
        # Если следующая позиция уже занята телом змейки,
        if next_position in self.body:
            # то удаляем из тела все элементы, начиная с этой позиции.
            self.body = self.body[:self.body.index(next_position)]
        elif next_position in field.foods:
            self.eat()
            field.foods.remove(next_position)
            field.add_food(field.random_food())
        # Если следующая позиция свободна,
        # то обновляем текущую позицию и направление движения.
        self.position = next_position
        self.direction = new_direction

        # Добавляем новую позицию в начало тела змейки.
        self.body.insert(0, self.position)
        # Если длина тела больше 1, то удаляем последний элемент тела.
        if len(self.body) > 1:
            self.body.pop() 

        print(self.look_around(self.VIEW_RADIUS))


    def change_direction(self, direction):
        if direction == "FORWARD":
            new_direction = self.direction
        elif direction == "LEFT":
            new_direction = (self.direction - 1) % 4  # Поворот налево
        elif direction == "RIGHT":
            new_direction = (self.direction + 1) % 4  # Поворот направо

        return new_direction

    def get_direction(self, i, j):
        if i == 0 and j < 0:
            return self.DIRECTION_UP
        elif i > 0 and j == 0:
            return self.DIRECTION_RIGHT
        elif i == 0 and j > 0:
            return self.DIRECTION_DOWN
        elif i < 0 and j == 0:
            return self.DIRECTION_LEFT
        else:
            return None
        
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
                    object_type = "wall"
                elif pos in field.snake_set:
                    object_type = "snake"
                elif pos in field.food_set:
                    object_type = "food"
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


    def eat(self):
        self.body.append(self.body[-1])  # добавляем новый сегмент в конец тела змейки

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

# Инициализация pygame
pygame.init()
fps_counter = FPS()
screen = pygame.display.set_mode((WORLD_WIDTH, WORLD_HEIGHT))

# Создание змейки и поля
field = Field((WORLD_WIDTH, WORLD_HEIGHT))

for _ in range(20):
    field.add_food(field.random_food())

for _ in range(10):
    field.add_wall(field.random_wall())

for _ in range(1):
    field.add_snake(field.random_snake())

# Главный цикл игры
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

    # Очистка экрана
    screen.fill((0, 0, 0))

    # fps_counter.count()

    # for snake in field.snakes:
    #     snake.step()

    # Отрисовка змеек
    for snake in field.snakes:
        for segment in snake.body:
            pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(segment[0], segment[1], WORLD_SCALE_FACTOR, WORLD_SCALE_FACTOR))

    # Отрисовка еды
    for food in field.foods:
        pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(food[0], food[1], WORLD_SCALE_FACTOR, WORLD_SCALE_FACTOR))

    for food in field.food_set:
        pygame.draw.rect(screen, (0, 100, 255), pygame.Rect(food[0]+5, food[1]+5, WORLD_SCALE_FACTOR-10, WORLD_SCALE_FACTOR-10))

    # Отрисовка стен
    for wall in field.walls:
        pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(wall[0], wall[1], WORLD_SCALE_FACTOR, WORLD_SCALE_FACTOR))

    # Обновление экрана
    pygame.display.flip()

    # Задержка
    # pygame.time.delay(100)
