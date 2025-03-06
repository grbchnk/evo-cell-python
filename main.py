import pygame
import sys
import random
import numpy as np
import time
from collections import deque
import datetime
import os
import pickle

# Константы конфигурации
WORLD_WIDTH = 780
WORLD_HEIGHT = 600
WORLD_SCALE_FACTOR = 4
LOAD_GENE_FILES = 10
SAVE_LAST_SNAKE = 5

max_generation = 0
MUTATION_RATE = 0.001
MUTATION_SCALE = 0.1

# Инициализация Pygame и шрифта
pygame.init()
pygame.font.init()
font = pygame.font.SysFont("Arial", 14)


class Perceptron:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.layers = len(layer_sizes)
        self.weights = [np.random.uniform(-1, 1, (layer_sizes[i], layer_sizes[i-1]))
                        for i in range(1, self.layers)]

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
        with open(os.path.join(folder, filename + ".pkl"), 'wb') as f:
            pickle.dump(self.get_weights(), f)

    def load_weights(self, filename):
        with open(filename, 'rb') as f:
            weights = pickle.load(f)
        self.set_weights(weights)

    def mutate_weights(self, generation):
        scale = MUTATION_SCALE / (1 + generation * 0.1)
        for i in range(len(self.weights)):
            mutation_mask = np.random.rand(
                *self.weights[i].shape) < MUTATION_RATE
            self.weights[i] += mutation_mask * \
                np.random.normal(0, scale, self.weights[i].shape)

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
    VIEW_RADIUS = 8
    MAX_ENERGY = 30
    MAX_LENGTH = 8

    def __init__(self, position, direction, body, color=None):
        self.position = position
        self.direction = direction
        self.body = deque(body)
        self.body_set = set(body)
        self.energy = self.MAX_ENERGY
        self.brain = Perceptron([10, 2, 2, 2, 3])
        self.last_prediction = None
        self.color = color if color is not None else (random.randint(
            0, 255), random.randint(0, 255), random.randint(0, 255))
        self.generation = 0

    def step(self):
        self.energy -= 1
        if self.energy <= 0:
            self.energy = self.MAX_ENERGY
            if self.body:
                self.body_set.remove(self.body.pop())
            if not self.body:
                self.dead()
                return

        input_data = self.look_around(self.VIEW_RADIUS)
        prediction = self.brain.predict(input_data)
        actions = ["FORWARD", "LEFT", "RIGHT"]
        action = actions[np.argmax(prediction)]
        self.last_prediction = action
        self.move(action)

    def move(self, action):
        if len(self.body) >= self.MAX_LENGTH:
            self.reproduce()

        new_direction = self.change_direction(action)
        next_position = self.calculate_next_position(new_direction)

        # Проверка столкновения с телами других змей (исключая собственное тело)
        if next_position in field.snake_set - self.body_set:
            self.dead()
            return

        # Столкновение с собой или со стенами
        if next_position in self.body_set or next_position in field.wall_set:
            self.dead()
        elif next_position in field.food_set:
            self.eat()
            field.foods.remove(next_position)
            field.add_food(field.random_food())
            self.update_position(next_position, new_direction, grow=True)
        else:
            self.update_position(next_position, new_direction, grow=False)

    def calculate_next_position(self, new_direction):
        if new_direction == self.DIRECTION_UP:
            return (self.position[0], (self.position[1] - WORLD_SCALE_FACTOR) % WORLD_HEIGHT)
        elif new_direction == self.DIRECTION_RIGHT:
            return ((self.position[0] + WORLD_SCALE_FACTOR) % WORLD_WIDTH, self.position[1])
        elif new_direction == self.DIRECTION_DOWN:
            return (self.position[0], (self.position[1] + WORLD_SCALE_FACTOR) % WORLD_HEIGHT)
        elif new_direction == self.DIRECTION_LEFT:
            return ((self.position[0] - WORLD_SCALE_FACTOR) % WORLD_WIDTH, self.position[1])

    def update_position(self, next_position, new_direction, grow):
        self.position = next_position
        self.direction = new_direction
        self.body.appendleft(self.position)
        self.body_set.add(self.position)
        if not grow and len(self.body) > 1:
            tail = self.body.pop()
            self.body_set.remove(tail)

    def change_direction(self, action):
        if action == "FORWARD":
            return self.direction
        elif action == "LEFT":
            return (self.direction - 1) % 4
        elif action == "RIGHT":
            return (self.direction + 1) % 4

    def look_around(self, view_radius):
        def get_delta_pos(direction, distance):
            if direction == self.DIRECTION_UP:
                return (0, -distance)
            elif direction == self.DIRECTION_RIGHT:
                return (distance, 0)
            elif direction == self.DIRECTION_DOWN:
                return (0, distance)
            elif direction == self.DIRECTION_LEFT:
                return (-distance, 0)

        def get_object_type(pos):
            if pos in field.wall_set or pos in field.snake_set:
                return -1
            elif pos in field.food_set:
                return 3
            else:
                return 0

        directions = [(self.direction - 1) %
                      4, self.direction, (self.direction + 1) % 4]
        closest_objects = [0] * 5
        closest_distances = [0] * 5

        # Поиск ближайших объектов в основных и диагональных направлениях
        for distance in range(1, view_radius + 1):
            for idx, d in enumerate(directions):
                dx, dy = get_delta_pos(d, distance)
                pos = ((self.position[0] + dx * WORLD_SCALE_FACTOR) % WORLD_WIDTH,
                       (self.position[1] + dy * WORLD_SCALE_FACTOR) % WORLD_HEIGHT)
                if closest_objects[idx] == 0:
                    obj = get_object_type(pos)
                    if obj != 0:
                        closest_objects[idx] = obj
                        closest_distances[idx] = round(
                            1 - distance / view_radius, 2)
            # Диагональные направления
            for j in range(2):
                d1 = directions[j]
                d2 = directions[(j+1) % 3]
                dx = (get_delta_pos(d1, distance)[
                      0] + get_delta_pos(d2, distance)[0])
                dy = (get_delta_pos(d1, distance)[
                      1] + get_delta_pos(d2, distance)[1])
                pos = ((self.position[0] + dx * WORLD_SCALE_FACTOR) % WORLD_WIDTH,
                       (self.position[1] + dy * WORLD_SCALE_FACTOR) % WORLD_HEIGHT)
                index = j + 1
                if closest_objects[index] == 0:
                    obj = get_object_type(pos)
                    if obj != 0:
                        closest_objects[index] = obj
                        closest_distances[index] = round(
                            1 - distance / view_radius, 2)

        result = []
        for i in range(5):
            result.extend([closest_objects[i], closest_distances[i]])
        return result

    def reproduce(self):
        half_length = len(self.body) // 2
        body_list = list(self.body)
        new_body = list(reversed(body_list[half_length:]))
        tail_direction = self.get_direction(
            body_list[-2], body_list[-1]) if len(body_list) >= 2 else self.direction

        child = Snake(new_body[0], tail_direction, new_body, self.color)
        # Оставляем в родителе только первую половину
        self.body = deque(body_list[:half_length])
        self.body_set = set(self.body)

        child.brain.set_weights(self.brain.get_weights())
        child.brain.mutate_weights(self.generation)

        child.MAX_LENGTH = self.MAX_LENGTH
        self.generation += 1
        global max_generation
        if self.generation > max_generation:
            max_generation = self.generation
            filename = "gene_max-" + \
                str(self.generation) + datetime.datetime.now().strftime("-%f")
            self.brain.save_weights(filename)
            self.brain.save_weights_max(filename)
            print(f"New max generation: {max_generation}")

        field.add_snake(child)

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

    def eat(self):
        # Поведение при поедании можно задавать здесь
        pass

    def dead(self):
        if self in field.snakes:
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
        self.snake_set = {pos for snake in self.snakes for pos in snake.body}

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
            position = (random.randint(0, self.size[0] // WORLD_SCALE_FACTOR - 1) * WORLD_SCALE_FACTOR,
                        random.randint(0, self.size[1] // WORLD_SCALE_FACTOR - 1) * WORLD_SCALE_FACTOR)
            if position not in self.snake_set and position not in self.wall_set and position not in self.food_set:
                return position

    def random_wall(self):
        while True:
            position = (random.randint(0, self.size[0] // WORLD_SCALE_FACTOR - 1) * WORLD_SCALE_FACTOR,
                        random.randint(0, self.size[1] // WORLD_SCALE_FACTOR - 1) * WORLD_SCALE_FACTOR)
            if position not in self.snake_set and position not in self.food_set:
                return position

    def random_snake(self):
        while True:
            position = (random.randint(0, self.size[0] // WORLD_SCALE_FACTOR - 1) * WORLD_SCALE_FACTOR,
                        random.randint(0, self.size[1] // WORLD_SCALE_FACTOR - 1) * WORLD_SCALE_FACTOR)
            if position not in self.wall_set and position not in self.food_set:
                direction = random.choice([Snake.DIRECTION_UP, Snake.DIRECTION_RIGHT,
                                           Snake.DIRECTION_DOWN, Snake.DIRECTION_LEFT])
                return Snake(position, direction, [position])


def initialize_game():
    global field
    field = Field((WORLD_WIDTH, WORLD_HEIGHT))

    # Добавление еды
    for _ in range(1000):
        field.add_food(field.random_food())

    # Добавление стен
    for _ in range(1500):
        field.add_wall(field.random_wall())

    # Загрузка генов, если файлов достаточно
    files = [file for file in os.listdir('.') if file.endswith('.pkl')]
    if len(files) < LOAD_GENE_FILES:
        for _ in range(1000):
            field.add_snake(field.random_snake())
        return

    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    # Удаление старых файлов генов
    for file in files[LOAD_GENE_FILES:]:
        os.remove(file)

    for i in range(min(LOAD_GENE_FILES, len(files))):
        gen_snake = field.random_snake()
        gen_snake.brain.load_weights(files[i])
        field.add_snake(gen_snake)

    # Загрузка генов максимального поколения
    def load_latest_weights(folder):
        if not os.path.exists(folder):
            return
        files = os.listdir(folder)
        files.sort(key=lambda x: os.path.getmtime(os.path.join(folder, x)))
        for q, file in enumerate(files[:10]):
            max_snake = field.random_snake()
            max_snake.brain.load_weights(os.path.join(folder, file))
            shade = 105 + q * 6
            max_snake.color = (shade, shade, shade)
            field.add_snake(max_snake)
    load_latest_weights("max_generation")


screen = pygame.display.set_mode((WORLD_WIDTH, WORLD_HEIGHT))
clock = pygame.time.Clock()

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

    # Если змей мало, сохраняем гены и перезапускаем игру
    if len(field.snakes) <= SAVE_LAST_SNAKE:
        for snake in field.snakes:
            snake.brain.save_weights(
                "gene-" + datetime.datetime.now().strftime("%f"))
        initialize_game()

    screen.fill((0, 0, 0))
    for snake in field.snakes[:]:
        snake.step()
    field.update_sets()

    # Отрисовка змей
    for snake in field.snakes:
        for segment in snake.body:
            pygame.draw.rect(screen, snake.color,
                             pygame.Rect(segment[0], segment[1], WORLD_SCALE_FACTOR, WORLD_SCALE_FACTOR))
    # Отрисовка еды
    for food in field.foods:
        pygame.draw.circle(screen, (0, 255, 0),
                           (food[0] + WORLD_SCALE_FACTOR//2, food[1] + WORLD_SCALE_FACTOR//2), int(WORLD_SCALE_FACTOR/1.2))
        pygame.draw.circle(screen, (255, 255, 255),
                           (food[0] + WORLD_SCALE_FACTOR//2, food[1] + WORLD_SCALE_FACTOR//2), int(WORLD_SCALE_FACTOR/2))
    # Отрисовка стен
    for wall in field.walls:
        pygame.draw.rect(screen, (100, 100, 100),
                         pygame.Rect(wall[0], wall[1], WORLD_SCALE_FACTOR, WORLD_SCALE_FACTOR))

    # Отрисовка информационной панели с количеством змей по цвету
    snake_counts_by_color = {}
    for snake in field.snakes:
        snake_counts_by_color[snake.color] = snake_counts_by_color.get(
            snake.color, 0) + 1
    total_snakes = sum(snake_counts_by_color.values())
    scaling_factor = WORLD_WIDTH / total_snakes if total_snakes > 0 else 0
    x = 0
    bar_height = 30
    for color, count in snake_counts_by_color.items():
        width = count * scaling_factor
        pygame.draw.rect(screen, (50, 50, 50), pygame.Rect(
            x, WORLD_HEIGHT - bar_height, width, bar_height))
        pygame.draw.rect(screen, color, pygame.Rect(
            x + 2, WORLD_HEIGHT - bar_height + 2, width - 4, bar_height - 4))
        text = font.render(str(count), True, (255, 255, 255))
        text_rect = text.get_rect(
            center=(x + width / 2, WORLD_HEIGHT - bar_height / 2))
        screen.blit(text, text_rect)
        x += width

    pygame.display.flip()
    clock.tick(60)  # Ограничение до 60 FPS
