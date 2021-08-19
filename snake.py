import sys
import pygame
import random
import math
import numpy as np
import snake_brain


# Difficulty settings
speed = 5

# Window size
SIZE = 10
PIX_SIZE = 50
FRAME_SIZE = SIZE * PIX_SIZE

# Checks for errors encountered
check_errors = pygame.init()
# pygame.init() example output -> (6, 0)
# second number in tuple gives number of errors
if check_errors[1] > 0:
    print(f'[!] Had {check_errors[1]} errors when initialising game, exiting...')
    sys.exit(-1)
else:
    print('[+] Game successfully initialised')


# Initialise game window
pygame.display.set_caption('Snake Eater')
game_window = pygame.display.set_mode((FRAME_SIZE, FRAME_SIZE))


# Colors (R, G, B)
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)


# FPS (frames per second) controller
fps_controller = pygame.time.Clock()



class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, point):
        return math.sqrt((point.x - self.x) ** 2 + (point.y - self.y) ** 2)

    # This method returns the angle of the line between self and obj in degrees between 0 and 360
    def angle(self, obj):
        # Return degrees between 0 and 360
        return math.degrees(math.atan2(obj.y - self.y, obj.x - self.x)) % 360

    def __eq__(self, obj):
        return isinstance(obj, Point) and obj.x == self.x and obj.y == self.y

    def __str__(self):
        return f'Point: ({self.x}, {self.y})'

class Snake:
    direction = 0

    def __init__(self):
        self.head = Point(5,4)
        self.body = [self.head, Point(5,5), Point(5,6), Point(5,7)]
        self.locate_food()
        self.direction = 0
        self.board = self.create_board()

    def create_board(self):
        # Create board
        board = []
        for i in range(SIZE):
            for j in range(SIZE):
                board.append(Point(i,j))
        return board

    def move(self, dir):
        # Store prevoius head position
        self.previous_head = self.head
        # Avoid forbidden directions
        if abs(self.direction - dir) != 2:
            self.direction = dir
        else:
            return
        # Moving process, head to new position
        if self.direction == 0:
            self.head = Point(self.body[0].x, self.body[0].y - 1)
        elif self.direction == 1:
            self.head = Point(self.body[0].x + 1, self.body[0].y)
        elif self.direction == 2:
            self.head = Point(self.body[0].x, self.body[0].y + 1)
        elif self.direction == 3:
            self.head = Point(self.body[0].x - 1, self.body[0].y)
        # Check if 
        if not self.has_eaten():
            self.body.pop()
        self.body.insert(0, self.head)

    def has_eaten(self):
        if self.head.x == self.food.x and self.head.y == self.food.y:
            self.locate_food()
            return True

    def locate_food(self):
        placed = False
        while not placed:
            food = Point(random.randint(0,SIZE-1), random.randint(0,SIZE-1))
            if food not in self.body and food != self.head:
                placed = True
        self.food = food

    def has_died(self):
        # If snake has no moved, model has to improve, so we kill the snake
        if self.previous_head == self.head:
            return True
        # Check if snake hits itself
        if self.head in self.body[1:]:
            return True
        # Check if snake hits board edge
        if self.head not in self.board:
            return True

    def draw(self):
        # Draw background
        game_window.fill(black)
        # Draw head
        pygame.draw.rect(game_window, red, pygame.Rect(self.head.x*PIX_SIZE, self.head.y*PIX_SIZE, PIX_SIZE-1, PIX_SIZE-1))
        # Draw body
        for b in self.body[1:]:
            pygame.draw.rect(game_window, green, pygame.Rect(b.x*PIX_SIZE, b.y*PIX_SIZE, PIX_SIZE-1, PIX_SIZE-1))
        # Draw food
        pygame.draw.rect(game_window, blue, pygame.Rect(self.food.x*PIX_SIZE, self.food.y*PIX_SIZE, PIX_SIZE-1, PIX_SIZE-1))
        # Refresh game screen
        pygame.display.update()

    def info(self):
        # Distance to board edge
        top = self.head.y
        right = SIZE - self.head.x - 1
        bottom = SIZE - self.head.y - 1
        left = self.head.x
        # Nomalize data
        top = self.norm(top, SIZE, 0)
        right = self.norm(right, SIZE, 0)
        bottom = self.norm(bottom, SIZE, 0)
        left = self.norm(left, SIZE, 0)
        print(f'{top=} {right=} {bottom=} {left=}')
        # Distance and angle to food
        distance = self.head.distance(self.food)
        angle = self.head.angle(self.food)
        # Normalize data
        max = math.sqrt(SIZE ** 2 + SIZE ** 2)
        distance = self.norm(distance, max, 0)
        angle = self.norm(angle, 360, 0)
        print(f'distance=%.2f {angle=}' % distance)
        # Distance to itself in 8 directions
        d1, d2, d3, d4, d5, d6, d7, d8 = self.distances()
        print(f'd1=%.2f d2=%.2f d3=%.2f d4=%.2f d5=%.2f d6=%.2f d7=%.2f d8=%.2f\n' % (d1, d2, d3, d4, d5, d6, d7, d8))
        data = np.array([[top, right, bottom, left, distance, angle, d1, d2, d3, d4, d5, d6, d7, d8]], dtype=np.float16)
        return data

    def distances(self):
        ''' Returns a list with distances from head to body or board limit
        distances = [top, top-left, left, down-left, down, down-right, right, top-right] '''
        directions_x_y = [[0,-1], [1,-1], [1,0], [1,1], [0,1], [-1,1], [-1,0], [-1,-1]]
        distances = []
        max = math.sqrt(SIZE ** 2 + SIZE ** 2)
        min = 0
        for x,y in directions_x_y:
            distance = 0
            test = Point(self.head.x + x, self.head.y + y)
            while test in self.board and test not in self.body[1:]:
                test.x += x
                test.y += y
                distance += 1
            # Diagonal distance
            if x and y:
                distances.append(self.norm(math.sqrt(distance**2 + distance**2), max, min))
            else:
                distances.append(self.norm(distance, max, min))
        return distances


    # This function normalizes the input data, between 0 and 1, given max and min values
    def norm(self, data, max, min):
        return (data - min) / (max - min)






# Main logic
snake = Snake()

while True:
    pressed_key = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        # Whenever a key is pressed down
        elif event.type == pygame.KEYDOWN:
            pressed_key = True
            # W -> Up; S -> Down; A -> Left; D -> Right
            # if event.key == pygame.K_UP or event.key == ord('w'):
            #     dir = 0
            # if event.key == pygame.K_RIGHT or event.key == ord('d'):
            #     dir = 1
            # if event.key == pygame.K_DOWN or event.key == ord('s'):
            #     dir = 2
            # if event.key == pygame.K_LEFT or event.key == ord('a'):
            #     dir = 3

            # Esc -> Create event to quit the game
            if event.key == pygame.K_ESCAPE:
                pygame.event.post(pygame.event.Event(pygame.QUIT))


    # Draw everything
    snake.draw()
    # Brain logic
    data = snake.info()
    desition = snake_brain.decide(data)
    snake.move(desition)
    # Check if snake has died
    if snake.has_died():
        w = snake_brain.model.get_weights()
        # Store the weights
        # snake_brain.model.save_weights('snake_brain.h5')
        snake_brain.model.set_weights(w)
        exit()

    # Refresh rate
    fps_controller.tick(speed)