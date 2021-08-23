from brain import Brain
import math
import random
import numpy as np



# Game config
SIZE = 15 # Stands for 'n' x 'n' board


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
    movements = 0
    movements_without_eat = 0
    turns = 0
    reproducible = True

    def __init__(self):
        self.brain = Brain()
        self.head = Point(SIZE//2, SIZE//2)
        self.set_body()
        self.set_board()
        self.locate_food()

    def set_body(self):
        self.body = []
        directions = [[0,1, 0], [1,0, 3], [0,-1, 2], [-1,0, 1]]
        x, y, self.direction = random.choice(directions)
        for i in range(1, 5):
            self.body.append(Point(self.head.x + (i*x), self.head.y + (i*y)))

    def set_board(self):
        self.board = []
        for i in range(SIZE):
            for j in range(SIZE):
                self.board.append(Point(i,j))

    def move(self, dir):
        # Store prevoius head position
        self.previous_head = self.head
        self.movements += 1
        self.movements_without_eat += 1
        # If direction has changed
        if dir != self.direction:
            self.turns += 1
        # Avoid forbidden directions
        if abs(self.direction - dir) != 2:
            self.direction = dir
        else:
            return
        # Moving process, head to new position
        if self.direction == 0:
            self.head = Point(self.head.x, self.head.y - 1)
        elif self.direction == 1:
            self.head = Point(self.head.x + 1, self.head.y)
        elif self.direction == 2:
            self.head = Point(self.head.x, self.head.y + 1)
        elif self.direction == 3:
            self.head = Point(self.head.x - 1, self.head.y)
        # Check if snake has found food
        if not self.has_eaten():
            self.body.pop()
        self.body.insert(0, self.previous_head)


    def has_eaten(self):
        if self.head.x == self.food.x and self.head.y == self.food.y:
            self.locate_food()
            self.moves_without_eat = 0
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
        # If snake is taking too long to feed, kill the snake
        if self.movements_without_eat > 50:
            self.reproducible = False
            return True
        # Check if snake hits itself
        if self.head in self.body:
            return True
        # Check if snake hits board edge
        if self.head not in self.board:
            return True

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
        # print(f'{top=} {right=} {bottom=} {left=}')
        # Distance and angle to food
        distance = self.head.distance(self.food)
        angle = self.head.angle(self.food)
        # Normalize data
        max = math.sqrt(SIZE ** 2 + SIZE ** 2)
        distance = self.norm(distance, max, 0)
        # angle = self.norm(angle, 360, 0)
        # print(f'distance=%.2f {angle=}' % distance)
        # Distance to itself in 8 directions
        d1, d2, d3, d4, d5, d6, d7, d8 = self.distances()
        # print(f'd1=%.2f d2=%.2f d3=%.2f d4=%.2f d5=%.2f d6=%.2f d7=%.2f d8=%.2f\n' % (d1, d2, d3, d4, d5, d6, d7, d8))
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

    def score(self):
        if not self.reproducible:
            return 0
        return len(self.body)*1 + self.movements*0.1 + self.turns * 0.02

    def reset(self):
        self.movements = 0
        self.turns = 0
        self.head = Point(SIZE//2, SIZE//2)
        self.set_body()
        self.locate_food()
        self.movements_without_eat = 0
        self.reproducible = True