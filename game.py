import sys
import pygame


# Game speed (frames per second)
speed = 15

# Window size
SIZE = 15 # Board size 'n' x 'n' boxes
PIX_SIZE = 25 # Size of box
FRAME_SIZE = SIZE * PIX_SIZE # Size of the game window

# Colors (R, G, B)
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)

# Initilized when game_start() is called
game_window = None
fps_controller = None


def game_start():
    ''' Creates the game window object '''
    global game_window, fps_controller
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

    # FPS (frames per second) controller
    fps_controller = pygame.time.Clock()



def draw(snake):
    ''' Receives a snake object and render it'''
    # Draw background
    game_window.fill(black)
    # Draw head
    pygame.draw.rect(game_window, red, pygame.Rect(snake.head.x*PIX_SIZE, snake.head.y*PIX_SIZE, PIX_SIZE-1, PIX_SIZE-1))
    # Draw body
    for b in snake.body:
        pygame.draw.rect(game_window, green, pygame.Rect(b.x*PIX_SIZE, b.y*PIX_SIZE, PIX_SIZE-1, PIX_SIZE-1))
    # Draw food
    pygame.draw.rect(game_window, blue, pygame.Rect(snake.food.x*PIX_SIZE, snake.food.y*PIX_SIZE, PIX_SIZE-1, PIX_SIZE-1))
    # Refresh game screen
    pygame.display.update()



def listen_events():
    ''' Listen for key presses '''
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        # Whenever a key is pressed down
        elif event.type == pygame.KEYDOWN:
            # W -> Up; S -> Down; A -> Left; D -> Right
            if event.key == pygame.K_UP or event.key == ord('w'):
                dir = 0
            if event.key == pygame.K_RIGHT or event.key == ord('d'):
                dir = 1
            if event.key == pygame.K_DOWN or event.key == ord('s'):
                dir = 2
            if event.key == pygame.K_LEFT or event.key == ord('a'):
                dir = 3

            # Esc -> Create event to quit the game
            if event.key == pygame.K_ESCAPE:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
    # Refresh rate
    fps_controller.tick(speed)