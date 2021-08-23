from snake import Snake
from datetime import datetime
from multiprocessing import Pool
from game import draw, listen_events, game_start



# Genetic algorithm config
POPULATION = 200
SURVIVAL_RATE = 0.2
MUTATION_RATE = 0.1
GENERATIONS = 5

# Multiprocessing config
PARALLEL = True





def play_game(snake, show=False):
    snake.reset()
    while True:
        # Brain logic
        if show:
            draw(snake)
            listen_events()
        data = snake.info()
        desition = snake.brain.decide(data)
        snake.move(desition)
        # Check if snake has died
        if snake.has_died():
            if PARALLEL:
                return snake
            else:
                break



if __name__ == '__main__':
    if PARALLEL:
        pool = Pool(processes=4)

    # Initialize snake population
    snakes = [Snake() for _ in range(POPULATION)]
    
    # Run simulation
    for g in range(GENERATIONS):

        time_start = datetime.now()
        print(f'Generation {g} starting at {time_start.strftime("%H:%M:%S")}')

        if PARALLEL:
            pass
            snakes = pool.map(play_game, snakes)
        else:
            for snake in snakes:
                play_game(snake, False)

        
        # Sort list of snakes by score
        snakes.sort(key=lambda x: x.score(), reverse=True)
        print(f'Best snake has a score of {snakes[0].score()} with {snakes[0].movements} movements and a length of {len(snakes[0].body)}')
        
        # Select survivors

        # Randomly select survivors

        
        # Print elapsed time
        time_end = datetime.now()
        print(f'Generation {g} ended in {time_end.minute - time_start.minute} minutes, {time_end.second - time_start.second} second\n')



    # After training play best snake in pygame
    game_start()
    snake = snakes[0]
    for snake in snakes[:5]:
        play_game(snake, True)