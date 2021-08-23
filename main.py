from snake import Snake
from datetime import datetime
from multiprocessing import Pool
from game import draw, listen_events, game_start
import random
import numpy as np



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


def crossover(snake1, snake2):
    ''' Given 2 parents generates 2 children.
    '''
    genes1 = snake1.brain.get_flattened_weights()
    genes2 = snake2.brain.get_flattened_weights()

    split_index = random.rangendint(0, len(genes1)-1)

    child1_genes = genes1[:split_index] + genes2[split_index:]
    child2_genes = genes2[:split_index] + genes1[split_index:]

    child1 = Snake()
    child2 = Snake()

    child1.brain.set_weights(child1_genes)
    child2.brain.set_weights(child2_genes)

    return child1, child2


def mutate(snake):
    ''' Randomly modifies a random picked weight '''
    flattened = snake.brain.get_flattened_weights()
    random_index = random.randint(0, len(flattened)-1)
    flattened[random_index] = np.random.randn()
    snake.brain.set_weights(flattened)
    return snake



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
            snakes = pool.map(play_game, snakes)
        else:
            for snake in snakes:
                play_game(snake, False)

        
        # Sort list of snakes by score
        snakes.sort(key=lambda x: x.score(), reverse=True)
        print(f'Best snake has a score of {snakes[0].score()} with {snakes[0].movements} movements and a length of {len(snakes[0].body)}')
        
        # Select survivors
        n_survivors = int(SURVIVAL_RATE * POPULATION)
        survivors = snakes[:n_survivors]

        # Breed new population
        new_population = []
        for _ in range(POPULATION - n_survivors):
            parent1 = survivors[int(n_survivors * (1 - MUTATION_RATE))]
            parent2 = survivors[int(n_survivors * (1 - MUTATION_RATE))]
            child = parent1.breed(parent2)
            new_population.append(child)

        # Mutate new population
        for snake in new_population:
            snake.mutate(MUTATION_RATE)

        # Replace old population with new population
        snakes = survivors + new_population

        # Print elapsed time
        time_end = datetime.now()
        print(f'Generation {g} ended in {time_end.minute - time_start.minute} minutes, {time_end.second - time_start.second} second\n')



    # After training play best snake in pygame
    game_start()
    snake = snakes[0]
    for snake in snakes[:5]:
        play_game(snake, True)