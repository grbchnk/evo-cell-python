import pygame
import sys
import random
import numpy as np

WORLD_WIDTH = 640
WORLD_HEIGHT = 480
WORLD_SCALE_FACTOR = 10



class Snake:
    DIRECTION_UP = 0
    DIRECTION_RIGHT = 1
    DIRECTION_DOWN = 2
    DIRECTION_LEFT = 3
    MAX_ENERGY = 100

    def __init__(self, position, direction, body, energy=MAX_ENERGY):
        self.position = position
        self.direction = direction
        self.body = body
        self.energy = energy

    def move(self, direction):
        self.energy -= 1  # змейка тратит энергию на движение

        if self.energy <= 0:
            self.energy = self.MAX_ENERGY
            self.body.pop()  # если энергия заканчивается, змейка уменьшает свое тело на 1

        new_direction = self.change_direction(direction)

        # Если новое направление вверх и текущее направление не вниз,
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

        
        pygame.draw.line(screen,  (255, 255, 255), self.position, [next_position[0]*2,next_position[1]*2], 10)
        # Если следующая позиция уже занята телом змейки,
        if next_position in self.body:
            # то удаляем из тела все элементы, начиная с этой позиции.
            self.body = self.body[:self.body.index(next_position)]
        elif next_position in field.food:
            self.eat()
            field.food.remove(next_position)
            field.food.append(field.random_food(snake))
        # Если следующая позиция свободна,
        # то обновляем текущую позицию и направление движения.
        self.position = next_position
        self.direction = new_direction
        print(self.body)
        # Добавляем новую позицию в начало тела змейки.
        self.body.insert(0, self.position)
        # Если длина тела больше 1, то удаляем последний элемент тела.
        if len(self.body) > 1:
            self.body.pop() 

    def change_direction(self, direction):
        if direction == "FORWARD":
            new_direction = self.direction
        elif direction == "LEFT":
            new_direction = (self.direction - 1) % 4  # Поворот налево
        elif direction == "RIGHT":
            new_direction = (self.direction + 1) % 4  # Поворот направо

        return new_direction
        

    def eat(self):
        self.body.append(self.body[-1])  # добавляем новый сегмент в конец тела змейки

class Field:
    def __init__(self, size, food, wall):
        self.size = size
        self.food = food
        self.wall = wall

    def random_food(self, snake):
        while True:
            position = (random.randint(0, self.size[0]//WORLD_SCALE_FACTOR - 1)*WORLD_SCALE_FACTOR, random.randint(0, self.size[1]//WORLD_SCALE_FACTOR - 1)*WORLD_SCALE_FACTOR)
            if position not in snake.body and position not in self.wall:
                return position

    def random_wall(self, snake):
        while True:
            position = (random.randint(0, self.size[0]//WORLD_SCALE_FACTOR - 1)*WORLD_SCALE_FACTOR, random.randint(0, self.size[1]//WORLD_SCALE_FACTOR - 1)*WORLD_SCALE_FACTOR)
            if position not in snake.body and position not in self.food:
                return position

# Инициализация pygame
pygame.init()

screen = pygame.display.set_mode((WORLD_WIDTH, WORLD_HEIGHT))

# Создание змейки и поля
snake = Snake((WORLD_WIDTH/2, WORLD_HEIGHT/2), Snake.DIRECTION_UP, [(WORLD_WIDTH/2, WORLD_HEIGHT/2)])
field = Field((WORLD_WIDTH,WORLD_HEIGHT), [], [])

# Add multiple food items to the field
for _ in range(50):
    field.food.append(field.random_food(snake))

# Add multiple wall items to the field
for _ in range(100):  # Change this number to spawn more or less walls
    field.wall.append(field.random_wall(snake))

# Главный цикл игры
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                snake.move("FORWARD")
            elif event.key == pygame.K_RIGHT:
                snake.move("RIGHT")
            elif event.key == pygame.K_LEFT:
                snake.move("LEFT")

    # Очистка экрана
    screen.fill((0, 0, 0))

    # Отрисовка змейки
    for segment in snake.body:
        pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(segment[0], segment[1], WORLD_SCALE_FACTOR, WORLD_SCALE_FACTOR))

    # Отрисовка еды
    for food in field.food:
        pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(food[0], food[1], WORLD_SCALE_FACTOR, WORLD_SCALE_FACTOR))

    # Отрисовка стен
    for wall in field.wall:
        pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(wall[0], wall[1], WORLD_SCALE_FACTOR, WORLD_SCALE_FACTOR))

    # Обновление экрана
    pygame.display.flip()

    # Задержка
    # pygame.time.delay(100)

