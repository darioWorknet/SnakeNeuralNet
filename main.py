from snake import Snake
from datetime import datetime
from multiprocessing import Pool
from game import draw, listen_events, game_start, wait_until_press_enter
import random
import numpy as np
import torch # NOT NEEDED, ONLY FOR TESTING
import time # NO NEED FOR THIS FUNCTION


# Genetic algorithm hyperparameters
POPULATION = 2000
SURVIVAL_RATE = 0.2
MUTATION_RATE = 0.1
GENERATIONS = 20

# Multiprocessing config
PARALLEL = True
N_PROCESSES = 3

# Show entire simulation in Pygame (only for single process)
SHOW = False
if PARALLEL:
    SHOW = False
if SHOW:
    game_start()


# Play the game with a single snake
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

    split_index = random.randrange(0, len(genes1) - 1)

    child1_genes = torch.cat((genes1[:split_index], genes2[split_index:]))
    child2_genes = torch.cat((genes2[:split_index], genes1[split_index:]))

    child1 = Snake()
    child2 = Snake()

    child1.brain.set_weights(child1_genes)
    child2.brain.set_weights(child2_genes)

    return child1, child2


def between_hiperline(snake1, snake2):
    # Genes 1 and 2 could be considered as hiperpoints
    genes1 = snake1.brain.get_flattened_weights()
    genes2 = snake2.brain.get_flattened_weights()
    # Summation of 2 torch vectors
    sum_genes = genes1 + genes2
    # Point between the 2 hyperpoints
    # Using div we get a point somewhere in the hiperline
    div = random.uniform(1.4, 2.9)
    child_between = sum_genes / div
    # New snake with those genes
    new_snake = Snake()
    new_snake.brain.set_weights(child_between)
    return new_snake



def mutate(snake):
    ''' Randomly modifies a randomly picked weight '''
    flattened = snake.brain.get_flattened_weights()
    random_index = random.randint(0, len(flattened)-1)
    flattened[random_index] = np.random.randn()
    snake.brain.set_weights(flattened)
    return snake


def return_random_snake(snakes):
    ''' Given a list of snakes returns a random one.
    Higher score snakes are more likely to be selected '''
    total_score = sum([x.get_score() for x in snakes])
    random_int = random.randint(0, int(total_score)-1)
    cumulative_score = 0
    for snake in snakes:
        cumulative_score += snake.get_score()
        if cumulative_score >= random_int:
            return snake
    return snakes[0]





if __name__ == '__main__':
    # Create pool object if we are using multiprocessing
    if PARALLEL:
        pool = Pool(processes=N_PROCESSES)

    # Create a list with best snakes, to be displayed at the end
    best_snakes = []
    appended_snakes = 0
    max_score = 30

    # Initialize snake population
    snakes = [Snake() for _ in range(POPULATION)]
    
    # Run simulation
    for g in range(GENERATIONS):

        time_start = datetime.now()
        print(f'Generation {g} starting at {time_start.strftime("%H:%M:%S")}')

        # Run the simulation (single process) or parallel (multiprocessing)
        if PARALLEL:
            snakes = pool.map(play_game, snakes)
        else:
            for snake in snakes:
                play_game(snake, SHOW)

        # Sort list of snakes by score
        snakes.sort(key=lambda x: x.get_score(), reverse=True)
        print(f'Best snake score: %0.2f   movements: {snakes[0].movements}   length: {len(snakes[0].body)}' % snakes[0].get_score())
        if snakes[0].get_score() > max_score:
            max_score = snakes[0].get_score()
            print(f"New snake appended, {appended_snakes} in total")
            appended_snakes += 1
            best_snakes.append(snakes[0])
        
        # Select survivors
        n_survivors = int(SURVIVAL_RATE * POPULATION)
        survivors = snakes[:n_survivors]
        
        # Breed new population
        new_population = []
        while len(new_population) <= POPULATION - n_survivors:
            # parents = random.sample(survivors, 2) # Random selection
            # Select 2 random parents (parents with more score are more likely to be selected)
            parent1 = return_random_snake(survivors)
            parent2 = return_random_snake(survivors)
            # Crossover parents
            childs = crossover(parent1, parent2)
            new_population.extend(childs)
            parent1 = return_random_snake(snakes)
            parent2 = return_random_snake(snakes)
            new_population.append(between_hiperline(parent1, parent2))

        # Mutate new population
        for snake in new_population:
            if np.random.uniform() < MUTATION_RATE:
                mutate(snake)

        # Replace old population with new population
        snakes = survivors + new_population

        # Print elapsed time
        time_end = datetime.now()
        print(f'Generation {g} ended in %.2f seconds\n' % (time_end - time_start).total_seconds())



    # After training play best snakes in pygame
    game_start()
    time.sleep(1)
    wait_until_press_enter()
    n_snakes = 20
    for snake in best_snakes: #snakes[:n_snakes]:
        play_game(snake, True)
    wait_until_press_enter()
    for snake in snakes[:30]: #snakes[:n_snakes]:
        play_game(snake, True)