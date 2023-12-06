import pygame
import sys
import random
import numpy as np
import time
from collections import deque
import datetime
import os
import pickle

WORLD_WIDTH = 780
WORLD_HEIGHT = 600
WORLD_SCALE_FACTOR = 4
LOAD_GENE_FILES = 10
SAVE_LAST_SNAKE = 5

max_generation = 0

MUTATION_RATE = 0.001  # вероятность мутации каждого веса
MUTATION_SCALE = 0.1 # масштаб (стандартное отклонение) мутаций

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
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.layers = len(layer_sizes)
        self.weights = [np.random.uniform(-1, 1, (layer_sizes[i], layer_sizes[i-1])) for i in range(1, self.layers)]
        
    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def print_weights(self):
        for i, w in enumerate(self.weights):
            print(f"Layer {i+1} weights: ", w)

    def save_weights(self, filename):
        with open(filename + ".pkl", 'wb') as f:
            pickle.dump(self.get_weights(), f)
    
    def save_weights_max(self, filename):
        folder = "max_generation"
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(folder + "/" + filename + ".pkl", 'wb') as f:
            pickle.dump(self.get_weights(), f)

    def load_weights(self, filename):
        with open(filename, 'rb') as f:
            weights = pickle.load(f)
        self.set_weights(weights)

    def mutate_weights(self):
        for i in range(len(self.weights)):
            self.weights[i] += (np.random.rand(*self.weights[i].shape) < MUTATION_RATE) * np.random.normal(0, MUTATION_SCALE, self.weights[i].shape)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x):
        return np.tanh(x)

    def predict(self, input_data):
        activations = input_data
        for weights in self.weights:
            outputs = np.dot(activations, weights.T)
            activations = self.tanh(outputs)
        return activations


class Snake:
    DIRECTION_UP = 0
    DIRECTION_RIGHT = 1
    DIRECTION_DOWN = 2
    DIRECTION_LEFT = 3
    VIEW_RADIUS = 16
    MAX_ENERGY = 30
    MAX_LENGTH = 8

    def __init__(self, position, direction, body, color=None):
        self.position = position
        self.direction = direction
        self.body = deque(body)
        self.body_set = set(body)
        self.energy = self.MAX_ENERGY
        self.brain = Perceptron([10, 2, 2, 2, 3])
        self.last_prediction = 0
        if color is None:
            self.color = (random.uniform(0, 255), random.uniform(0, 255), random.uniform(0, 255))
        else:
            self.color = color
        self.generation = 0

    def step(self):
        self.energy -= 1
        
        if self.energy <= 0:
            self.energy = self.MAX_ENERGY
            self.body_set.remove(self.body.pop())

        if len(self.body) > 0:
            input_data = self.look_around(self.VIEW_RADIUS)

            prediction = self.brain.predict(input_data)
            self.last_prediction = prediction
            actions = ["FORWARD", "LEFT", "RIGHT"]
            
            action = actions[np.argmax(prediction)]
            self.last_prediction = action
            snake.move(action)
        else:
            self.dead()

    def move(self, direction):
        if len(self.body) >= self.MAX_LENGTH:
            self.reproduce()
        
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
            self.dead()
            # print("Dead in self" + str(self.look_around(self.VIEW_RADIUS)))
            # print("Choice: " + self.last_prediction)
            # то удаляем из тела все элементы, начиная с этой позиции.
            # while self.body[-1] != next_position:
            #     self.body_set.remove(self.body.pop())
            # self.body_set.remove(self.body.pop())
        elif next_position in field.wall_set:
            self.dead()
            # print("Dead in wall" + str(self.look_around(self.VIEW_RADIUS)))
            # print("Choice: " + self.last_prediction)

        elif next_position in field.food_set:
            self.eat()
            field.foods.remove(next_position)
            field.add_food(field.random_food())

            self.position = next_position
            self.direction = new_direction

            self.body.appendleft(self.position)
            self.body_set.add(self.position)
        else:
            self.position = next_position
            self.direction = new_direction

            self.body.appendleft(self.position)
            self.body_set.add(self.position)

            if len(self.body) > 1:
                self.body_set.remove(self.body.pop())

    def get_direction(self, pos1, pos2):
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
            new_direction = (self.direction - 1) % 4
        elif direction == "RIGHT":
            new_direction = (self.direction + 1) % 4

        return new_direction

    def look_around(self, view_radius):
        def get_delta_pos(direction, distance):
            if direction == self.DIRECTION_UP:
                return 0, -distance
            elif direction == self.DIRECTION_RIGHT:
                return distance, 0
            elif direction == self.DIRECTION_DOWN:
                return 0, distance
            elif direction == self.DIRECTION_LEFT:
                return -distance, 0

        def get_object_type(pos):
            if pos in field.wall_set or pos in field.snake_set:
                return -1
            elif pos in field.food_set:
                return 3
            else:
                return 0

        directions = [(self.direction - 1) % 4, self.direction, (self.direction + 1) % 4]
        closest_objects = [0] * 5
        closest_distances = [0] * 5
        result = [0] * 10

        for distance in range(1, view_radius + 1):
            for i, direction in enumerate(directions):
                dx, dy = get_delta_pos(direction, distance)
                pos = ((self.position[0] + dx * WORLD_SCALE_FACTOR) % WORLD_WIDTH, (self.position[1] + dy * WORLD_SCALE_FACTOR) % WORLD_HEIGHT)
                object_type = get_object_type(pos)
                if object_type != 0 and closest_objects[i] == 0:
                    closest_objects[i] = object_type
                    closest_distances[i] = round(1 - distance / view_radius, 2)

            for j in range(2):
                dx1, dy1 = get_delta_pos(directions[j], distance)
                dx2, dy2 = get_delta_pos(directions[(j+1)%3], distance)
                pos = ((self.position[0] + (dx1 + dx2) * WORLD_SCALE_FACTOR) % WORLD_WIDTH, (self.position[1] + (dy1 + dy2) * WORLD_SCALE_FACTOR) % WORLD_HEIGHT)
                object_type = get_object_type(pos)
                if object_type != 0 and closest_objects[i+j+1] == 0:
                    closest_objects[i+j+1] = object_type
                    closest_distances[i+j+1] = round(1 - distance / view_radius, 2)

        for k in range(5):
            result[k*2] = closest_objects[k]
            result[k*2+1] = closest_distances[k]

        return result

    def reproduce(self):
        half_length = len(self.body) // 2

        # вычисляем направление хвоста старой змейки
        tail_direction = self.get_direction(self.body[-2], self.body[-1])

        # создаем новую змейку с половиной тела старой змейки
        body_list = list(self.body)
        new_body = body_list[half_length:]
        new_body.reverse()  # разворачиваем список тела второй змейки

        child = Snake(new_body[0], tail_direction, new_body, self.color)

        # удаляем половину тела у старой змейки
        self.body = deque(body_list[:half_length])
        self.body_set = set(self.body)

        child.brain.set_weights(self.brain.get_weights())

        # мутация весов
        child.brain.mutate_weights()

        # self.MAX_LENGTH += 1
        child.MAX_LENGTH = self.MAX_LENGTH

        self.generation += 1
        global max_generation
        if self.generation > max_generation:
            max_generation = self.generation
            self.brain.save_weights("gene_max-" + str(self.generation) + datetime.datetime.now().strftime("-%f"))
            self.brain.save_weights_max("gene_max-" + str(self.generation) + datetime.datetime.now().strftime("-%f"))
            print(max_generation)

        field.add_snake(child)

    def eat(self):
        # шото тут было но уже нету
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
            if position not in [segment for snake in self.snakes for segment in snake.body] and position not in self.wall_set and position not in self.food_set:
                return position

    def random_wall(self):
        while True:
            position = (random.randint(0, self.size[0]//WORLD_SCALE_FACTOR - 1)*WORLD_SCALE_FACTOR, random.randint(0, self.size[1]//WORLD_SCALE_FACTOR - 1)*WORLD_SCALE_FACTOR)
            if position not in [segment for snake in self.snakes for segment in snake.body] and position not in self.food_set:
                return position

    def random_snake(self):
        while True:
            position = (random.randint(0, self.size[0]//WORLD_SCALE_FACTOR - 1)*WORLD_SCALE_FACTOR, random.randint(0, self.size[1]//WORLD_SCALE_FACTOR - 1)*WORLD_SCALE_FACTOR)
            if position not in self.wall_set and position not in self.food_set:
                return Snake(position, random.choice([Snake.DIRECTION_UP, Snake.DIRECTION_RIGHT, Snake.DIRECTION_DOWN, Snake.DIRECTION_LEFT]), [position])

# def generate_maze(width, height):
#     # Массив, который будет содержать данные лабиринта
#     maze = [['WALL' for x in range(width)] for y in range(height)]

#     # Стек для обхода в глубину
#     stack = []

#     # Выбираем случайную начальную точку
#     start_x = random.randint(0, width//2)*2
#     start_y = random.randint(0, height//2)*2

#     # Добавляем начальную точку в стек
#     stack.append((start_x, start_y))

#     # Пока стек не пуст
#     while stack:
#         x, y = stack.pop()

#         # Определяем направления прокладки пути
#         directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
#         random.shuffle(directions)

#         # Пытаемся проложить путь в каждом направлении
#         for dx, dy in directions:
#             nx, ny = x + dx*2, y + dy*2

#             # Если новая позиция находится в пределах лабиринта и является стеной
#             if (0 <= nx < width) and (0 <= ny < height) and maze[ny][nx] == 'WALL':
#                 # Прокладываем путь к новой позиции
#                 maze[y+dy][x+dx] = 'SPACE'
#                 maze[ny][nx] = 'SPACE'

#                 # Добавляем новую позицию в стек
#                 stack.append((nx, ny))

#     return maze



def initialize_game():
    global field
    field = Field((WORLD_WIDTH, WORLD_HEIGHT))
    # maze = generate_maze(WORLD_WIDTH//(WORLD_SCALE_FACTOR*3), WORLD_HEIGHT//(WORLD_SCALE_FACTOR*3))
    # for y in range(WORLD_HEIGHT//(WORLD_SCALE_FACTOR*3)):
    #     for x in range(WORLD_WIDTH//(WORLD_SCALE_FACTOR*3)):
    #         if maze[y][x] == 'WALL':
    #             xg = random.random()
    #             if xg < 0.3:
    #                 field.add_wall((x*(WORLD_SCALE_FACTOR*3), y*(WORLD_SCALE_FACTOR*3)))
    #                 field.add_wall((x*(WORLD_SCALE_FACTOR*3)-5, y*(WORLD_SCALE_FACTOR*3)))
    #                 field.add_wall((x*(WORLD_SCALE_FACTOR*3)+5, y*(WORLD_SCALE_FACTOR*3)))
    #             elif xg > 0.3 and xg < 0.6:
    #                 field.add_wall((x*(WORLD_SCALE_FACTOR*3), y*(WORLD_SCALE_FACTOR*3)))
    #                 field.add_wall((x*(WORLD_SCALE_FACTOR*3), y*(WORLD_SCALE_FACTOR*3)-5))
    #                 field.add_wall((x*(WORLD_SCALE_FACTOR*3), y*(WORLD_SCALE_FACTOR*3)+5))
    #             else:
    #                 field.add_wall((x*(WORLD_SCALE_FACTOR*3), y*(WORLD_SCALE_FACTOR*3)))
    #                 field.add_wall((x*(WORLD_SCALE_FACTOR*3), y*(WORLD_SCALE_FACTOR*3)-5))
    #                 field.add_wall((x*(WORLD_SCALE_FACTOR*3), y*(WORLD_SCALE_FACTOR*3)+5))
    #                 field.add_wall((x*(WORLD_SCALE_FACTOR*3)-5, y*(WORLD_SCALE_FACTOR*3)))
    #                 field.add_wall((x*(WORLD_SCALE_FACTOR*3)+5, y*(WORLD_SCALE_FACTOR*3)))

    for _ in range(1000):
        field.add_food(field.random_food())

    for _ in range(1500):
        field.add_wall(field.random_wall())

    # for _ in range(1000):
    #     field.add_snake(field.random_snake())

    for _ in range(2):
        def load_latest_weights(folder):
            # Получаем список всех файлов в папке
            files = os.listdir(folder)
            # Сортируем файлы по дате изменения (самые новые в конце)
            files.sort(key=lambda x: os.path.getmtime(folder + "/" + x))
            # Берем последний файл
            q = 0
            for file in files[:10]:
                max_snake = field.random_snake()
                
            # Загружаем веса из этого файла
                max_snake.brain.load_weights(folder + "/" + file)
                max_snake.color = (105 + q * 6, 105 + q * 6, 105 + q * 6)
                field.add_snake(max_snake)
                q+=1

        files = os.listdir('.')

        # Фильтруем список, чтобы оставить только файлы .pkl
        files = [file for file in files if file.endswith('.pkl')]

        if len(files) < LOAD_GENE_FILES:
            for _ in range(1000):
                field.add_snake(field.random_snake())
            return

        # Сортируем файлы по времени изменения (самый последний будет первым)
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        for file in files[LOAD_GENE_FILES:]:
            os.remove(file)

        for i in range(LOAD_GENE_FILES):
            last_file = files[i]
            gen_snake = field.random_snake()
            gen_snake.brain.load_weights(last_file)
            field.add_snake(gen_snake)

        load_latest_weights("max_generation")



pygame.init()
screen = pygame.display.set_mode((WORLD_WIDTH, WORLD_HEIGHT))
fps_counter = FPS()


snake_counts_by_color = {}

initialize_game()

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
            elif event.key == pygame.K_r:
                initialize_game()

    if len(field.snakes) <= SAVE_LAST_SNAKE:
        for snake in field.snakes:
            snake.brain.save_weights("gene-" + datetime.datetime.now().strftime("%f"))
        initialize_game()

    screen.fill((0, 0, 0))

    for snake in field.snakes:
        snake.step()

    # fps_counter.count()

    for snake in field.snakes:
        for segment in snake.body:
            pygame.draw.rect(screen, snake.color, pygame.Rect(segment[0], segment[1], WORLD_SCALE_FACTOR, WORLD_SCALE_FACTOR))

    for food in field.foods:
        pygame.draw.circle(screen, (0, 255, 0), (food[0]+WORLD_SCALE_FACTOR/2, food[1]+WORLD_SCALE_FACTOR/2), WORLD_SCALE_FACTOR/1.2)
        pygame.draw.circle(screen, (255, 255, 255), (food[0]+WORLD_SCALE_FACTOR/2, food[1]+WORLD_SCALE_FACTOR/2), WORLD_SCALE_FACTOR/2)

    for wall in field.walls:
        pygame.draw.rect(screen, (100, 100, 100), pygame.Rect(wall[0], wall[1], WORLD_SCALE_FACTOR, WORLD_SCALE_FACTOR))



    snake_counts_by_color = {}

    for snake in field.snakes:
        if snake.color not in snake_counts_by_color:
            snake_counts_by_color[snake.color] = 1
        else:
            snake_counts_by_color[snake.color] += 1

    total_snakes = sum(snake_counts_by_color.values())
    scaling_factor = WORLD_WIDTH / total_snakes

    x = 0
    for color, count in snake_counts_by_color.items():
        width = count * scaling_factor
        pygame.draw.rect(screen, color, pygame.Rect(x, WORLD_HEIGHT - 20, width, WORLD_HEIGHT))
        x += width



    pygame.display.flip()