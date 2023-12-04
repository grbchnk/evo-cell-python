import pygame
import sys
import random
import numpy as np
import time
from collections import deque
import matplotlib.pyplot as plt
import pickle
import datetime
import os

WORLD_WIDTH = 960
WORLD_HEIGHT = 640
WORLD_SCALE_FACTOR = 4

max_generation = 0

MUTATION_RATE = 0.001  # вероятность мутации каждого веса
MUTATION_SCALE = 0.05 # масштаб (стандартное отклонение) мутаций

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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_layer_weights = [[random.uniform(-1, 1) for _ in range(input_dim)] for _ in range(hidden_dim)]
        self.hidden_layer_weights = [[random.uniform(-1, 1) for _ in range(hidden_dim)] for _ in range(output_dim)]

    def get_weights(self):
        return self.input_layer_weights, self.hidden_layer_weights

    def set_weights(self, weights):
        self.input_layer_weights, self.hidden_layer_weights = weights

    def print_weights(self):
        print("Input layer weights: ", self.input_layer_weights)
        print("Hidden layer weights: ", self.hidden_layer_weights)

    def save_weights(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.get_weights(), f)

    def load_weights(self, filename):
        with open(filename, 'rb') as f:
            weights = pickle.load(f)
        self.set_weights(weights)

    def mutate_weights(self):
        for i in range(len(self.input_layer_weights)):
            for j in range(len(self.input_layer_weights[i])):
                if random.random() < MUTATION_RATE:
                    self.input_layer_weights[i][j] += random.gauss(0, MUTATION_SCALE)
        
        for i in range(len(self.hidden_layer_weights)):
            for j in range(len(self.hidden_layer_weights[i])):
                if random.random() < MUTATION_RATE:
                    self.hidden_layer_weights[i][j] += random.gauss(0, MUTATION_SCALE)


    def relu(self, x):
        return max(0, x)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def predict(self, x):
        hidden_layer_values = [sum(x*y for x, y in zip(x, weight)) for weight in self.input_layer_weights]
        hidden_layer_outputs = [self.relu(value) for value in hidden_layer_values]
        
        output_layer_values = [sum(ho*hw for ho, hw in zip(hidden_layer_outputs, weight)) for weight in self.hidden_layer_weights]
        output_layer_outputs = self.softmax(output_layer_values)

        return output_layer_outputs


class Snake:
    DIRECTION_UP = 0
    DIRECTION_RIGHT = 1
    DIRECTION_DOWN = 2
    DIRECTION_LEFT = 3
    VIEW_RADIUS = 16
    MAX_ENERGY = 30
    MAX_LENGTH = 4

    def __init__(self, position, direction, body, color=None):
        self.position = position
        self.direction = direction
        self.body = deque(body)
        self.body_set = set(body)
        self.energy = self.MAX_ENERGY
        self.brain = Perceptron(8, 8, 3)
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
            closest_objects, closest_distances = self.look_around(self.VIEW_RADIUS)
            input_data = closest_objects + closest_distances

            prediction = self.brain.predict(input_data)

            actions = ["FORWARD", "LEFT", "RIGHT"]
            action = actions[np.argmax(prediction)]

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

            self.body.appendleft(self.position)
            self.body_set.add(self.position)

            if len(self.body) > 1:
                self.body_set.remove(self.body.pop())

    def get_direction(self, pos1, pos2):
        # направление от pos1 к pos2
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
        directions = [(self.direction - 1) % 4, self.direction, (self.direction + 1) % 4]
        closest_objects = [0, 0, 0, 0, 0]
        closest_distances = [view_radius + 1, view_radius + 1, view_radius + 1, view_radius + 1, view_radius + 1]

        for distance in range(1, view_radius + 1):
            for i, direction in enumerate(directions):
                if direction == self.DIRECTION_UP:
                    pos = (self.position[0], (self.position[1] - distance * WORLD_SCALE_FACTOR) % WORLD_HEIGHT)
                elif direction == self.DIRECTION_RIGHT:
                    pos = ((self.position[0] + distance * WORLD_SCALE_FACTOR) % WORLD_WIDTH, self.position[1])
                elif direction == self.DIRECTION_DOWN:
                    pos = (self.position[0], (self.position[1] + distance * WORLD_SCALE_FACTOR) % WORLD_HEIGHT)
                elif direction == self.DIRECTION_LEFT:
                    pos = ((self.position[0] - distance * WORLD_SCALE_FACTOR) % WORLD_WIDTH, self.position[1])

                if pos in field.wall_set:
                    object_type = -1
                elif pos in field.snake_set:
                    object_type = -1
                elif pos in field.food_set:
                    object_type = 3
                else:
                    continue

                if distance < closest_distances[i]:
                    closest_objects[i] = object_type
                    closest_distances[i] = round(1 - distance / view_radius, 2)

        for distance in range(1, view_radius + 1):
            if self.direction == self.DIRECTION_UP:
                pos_diag1 = ((self.position[0] - distance * WORLD_SCALE_FACTOR) % WORLD_WIDTH, (self.position[1] - distance * WORLD_SCALE_FACTOR) % WORLD_HEIGHT)
                pos_diag2 = ((self.position[0] + distance * WORLD_SCALE_FACTOR) % WORLD_WIDTH, (self.position[1] - distance * WORLD_SCALE_FACTOR) % WORLD_HEIGHT)
            elif self.direction == self.DIRECTION_RIGHT:
                pos_diag1 = ((self.position[0] + distance * WORLD_SCALE_FACTOR) % WORLD_WIDTH, (self.position[1] + distance * WORLD_SCALE_FACTOR) % WORLD_HEIGHT)
                pos_diag2 = ((self.position[0] + distance * WORLD_SCALE_FACTOR) % WORLD_WIDTH, (self.position[1] - distance * WORLD_SCALE_FACTOR) % WORLD_HEIGHT)
            elif self.direction == self.DIRECTION_DOWN:
                pos_diag1 = ((self.position[0] + distance * WORLD_SCALE_FACTOR) % WORLD_WIDTH, (self.position[1] + distance * WORLD_SCALE_FACTOR) % WORLD_HEIGHT)
                pos_diag2 = ((self.position[0] - distance * WORLD_SCALE_FACTOR) % WORLD_WIDTH, (self.position[1] + distance * WORLD_SCALE_FACTOR) % WORLD_HEIGHT)
            elif self.direction == self.DIRECTION_LEFT:
                pos_diag1 = ((self.position[0] - distance * WORLD_SCALE_FACTOR) % WORLD_WIDTH, (self.position[1] - distance * WORLD_SCALE_FACTOR) % WORLD_HEIGHT)
                pos_diag2 = ((self.position[0] - distance * WORLD_SCALE_FACTOR) % WORLD_WIDTH, (self.position[1] + distance * WORLD_SCALE_FACTOR) % WORLD_HEIGHT)

            positions = [pos_diag1, pos_diag2]
            for i, pos in enumerate(positions):
                if pos in field.wall_set:
                    object_type = -1
                elif pos in field.snake_set:
                    object_type = -1
                elif pos in field.food_set:
                    object_type = 3
                else:
                    continue

                if distance < closest_distances[i+3]:
                    closest_objects[i+3] = object_type
                    closest_distances[i+3] = round(1 - distance / view_radius, 2)

        for k in range(5):
            if closest_objects[k] == 0:
                closest_distances[k] = 0

        return closest_objects, closest_distances

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

        child.MAX_LENGTH = self.MAX_LENGTH + 0.5

        child.generation = self.generation + 1
        global max_generation
        if child.generation > max_generation:
            max_generation = child.generation
            self.brain.save_weights("gene_max-" + str(child.generation) + "_" + datetime.datetime.now().strftime("-%f") + ".pkl")
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

def initialize_game():
    global field
    field = Field((WORLD_WIDTH, WORLD_HEIGHT))

    for _ in range(800):
        field.add_food(field.random_food())

    for _ in range(1000):
        field.add_wall(field.random_wall())

    for _ in range(2):
                # field.add_snake(field.random_snake())


        # files = os.listdir('.')

        # # Фильтруем список, чтобы оставить только файлы .pkl
        # files = [file for file in files if file.endswith('.pkl')]

        # # Сортируем файлы по времени изменения (самый последний будет первым)
        # files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        # # Берем первый файл из списка
        # last_file = files[0]

        # gen_snake = field.random_snake()
        # gen_snake.brain.load_weights(last_file)
        # field.add_snake(gen_snake)

        files = os.listdir('.')

        # Фильтруем список, чтобы оставить только файлы .pkl
        files = [file for file in files if file.endswith('.pkl')]

        # Сортируем файлы по времени изменения (самый последний будет первым)
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        for file in files[20:]:
            os.remove(file)

        for i in range(20):
            last_file = files[i]
            gen_snake = field.random_snake()
            gen_snake.brain.load_weights(last_file)
            field.add_snake(gen_snake)

pygame.init()
screen = pygame.display.set_mode((WORLD_WIDTH, WORLD_HEIGHT))
fps_counter = FPS()

snake_counts_by_color = {}
step_counter = 10
plt.ion()

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

    if len(field.snakes) <= 10:
        for snake in field.snakes:
            snake.brain.save_weights("gene_" + datetime.datetime.now().strftime("-%f") + ".pkl")
        initialize_game()

    screen.fill((0, 0, 0))

    for snake in field.snakes:
        snake.step()

    for snake in field.snakes:
        for segment in snake.body:
            pygame.draw.rect(screen, snake.color, pygame.Rect(segment[0], segment[1], WORLD_SCALE_FACTOR, WORLD_SCALE_FACTOR))

    for food in field.foods:
        pygame.draw.circle(screen, (0, 255, 0), (food[0]+WORLD_SCALE_FACTOR/2, food[1]+WORLD_SCALE_FACTOR/2), WORLD_SCALE_FACTOR/1.2)
        pygame.draw.circle(screen, (255, 255, 255), (food[0]+WORLD_SCALE_FACTOR/2, food[1]+WORLD_SCALE_FACTOR/2), WORLD_SCALE_FACTOR/2)

    for wall in field.walls:
        pygame.draw.rect(screen, (255, 0, 255), pygame.Rect(wall[0], wall[1], WORLD_SCALE_FACTOR, WORLD_SCALE_FACTOR))



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