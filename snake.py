from brain import Brain
import math
import random
import numpy as np

'''
Make a new system of reward snake each movement
    Getting close of food +1
    Getting close to walls -1


    Calculate:
        Distance to food: every step
        Distance to the wall in front: every step
'''

# Game config
SIZE = 15 # Stands for 'n' x 'n' board

DEBUG = False

def dprint(msg):
    if DEBUG:
        print(msg)


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

    def __init__(self, has_brain = False):
        if not has_brain:
            self.brain = Brain()
        self.head = Point(SIZE//2, SIZE//2)
        self.set_body()
        self.set_board()
        self.locate_food()
        self.movements = 0
        self.movements_without_eat = 0
        self.turns = 0
        self.reproducible = True
        self.score = 0.0

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
        self.previous_direction = self.direction
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
        movements = [[0,-1], [1,0], [0,1], [-1,0]]
        x, y = movements[self.direction]
        self.head = Point(self.head.x + x, self.head.y + y)
        # Check if snake has found food
        if not self.has_eaten():
            self.body.pop()
        self.body.insert(0, self.previous_head)
        self.update_score()


    def has_eaten(self):
        if self.head.x == self.food.x and self.head.y == self.food.y:
            self.locate_food()
            self.moves_without_eat = 0
            dprint("You eated food")
            self.score += 10
            return True

    def locate_food(self):
        while True:
            food = Point(random.randint(0,SIZE-1), random.randint(0,SIZE-1))
            if food not in self.body and food != self.head:
                self.food = food
                break

    def has_died(self):
        # If snake has no moved, model has to improve, so we kill the snake
        if self.previous_head == self.head:
            self.score -= 15
            return True
        # If snake is taking too long to feed, kill the snake
        if self.movements_without_eat > 50:
            self.score -= 10
            return True
        # Check if snake hits itself
        if self.head in self.body:
            return True
        # Check if snake hits board edge
        if self.head not in self.board:
            self.score -= 5
            return True
        

    def update_score(self):
        # Check if snake is getting closer to food
        prev_distance_to_food = self.previous_head.distance(self.food)
        curr_distance_to_food = self.head.distance(self.food)
        if curr_distance_to_food < prev_distance_to_food:
            dprint("Going closer to food")
            self.score += 1
            return
        # Check if snake is not getting closer to food
        # Check if is going closer of further from walls
        prev_distance_to_wall = self.distance_to_front_wall(self.previous_head, self.previous_direction)
        curr_distance_to_wall = self.distance_to_front_wall(self.head, self.direction)
        if curr_distance_to_wall > prev_distance_to_wall:
            dprint("Going further to walls")
            self.score -= 0.1
        else:
            dprint("Going closer to walls")
            self.score -= 0.2

    def distance_to_front_wall(self, head, direction):
        if direction == 0:
            return head.y
        elif direction == 1:
            return SIZE - head.x - 1
        elif direction == 2:
            return SIZE - head.y - 1
        elif direction == 3:
            return head.x


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
        dprint(f'{top=} {right=} {bottom=} {left=}')
        # Distance and angle to food
        distance = self.head.distance(self.food)
        angle = self.head.angle(self.food)
        # Normalize data
        max = math.sqrt(SIZE ** 2 + SIZE ** 2)
        distance = self.norm(distance, max, 0)
        # angle = self.norm(angle, 360, 0)
        dprint(f'distance=%.2f {angle=}' % distance)
        # Distance to itself in 8 directions
        d1, d2, d3, d4, d5, d6, d7, d8 = self.distances()
        dprint(f'd1=%.2f d2=%.2f d3=%.2f d4=%.2f d5=%.2f d6=%.2f d7=%.2f d8=%.2f\n' % (d1, d2, d3, d4, d5, d6, d7, d8))
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

    def get_score(self):
        # Prevent score to be 0 or below for further calculations
        if self.score < 1:
            return 1
        return self.score

    def reset(self):
        self.__init__(has_brain=True)