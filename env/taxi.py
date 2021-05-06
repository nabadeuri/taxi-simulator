import os
import pygame
import numpy as np
from env.settings import *
from env.sprites import *
from util import seeding
from util.colors import bcolors

# set the window position
os.environ["SDL_VIDEO_WINDOW_POS"] = "%d,%d" % (700, 80)

# initialize the pygame
pygame.init()

# set the window width and height
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

# fps
clock = pygame.time.Clock()

# a list for storing  taxi coordinates
taxi_coord = [[0 for x in range(NUM_ROWS)] for y in range(NUM_COLUMNS)]

# fill the coordinate for drawing taxi
x, y = 0, 0
for i in range(NUM_ROWS):
    for j in range(NUM_COLUMNS):
        taxi_coord[i][j] = (x, y)
        x += BOX_WIDTH
    x = 0
    y += BOX_HEIGHT

# coordinate of passenger and destination
passDesCoord = [
    (0, 0),
    (560, 0),
    (0, 560),
    (490, 560),
]

obstacleVertical = [
    (140, 0),  # topmid
    (140, 70),  # topmid
    (70, 350),  # bottom left
    (70, 420),  # bottom left
    (70, 490),  # bottom left
    (70, 560),  # bottom left
    (210, 70),  # center mid
    (210, 140),  # center mid
    (210, 210),  # center mid
    (210, 280),  # center mid
    (210, 350),  # center mid
    (210, 420),  # center mid
    (210, 490),  # center mid
    (490, 280),  # right bottom
    (490, 350),  # right bottom
    (490, 420),  # right bottom
    (490, 490),  # right bottom
    (490, 560),  # right bottom
]

obstacleHorizontal = [
    (210, 280),  # center mid
    (140, 280),  # center mid
    (0, 280),  # center mid
    (70, 280),  # center mid
    (560, 210),  # top right
    (490, 210),  # top right
    (420, 210),  # top right
    (350, 210),  # top right
    (280, 210),  # top right
]


# for randomly retrieve a state
def get_state(prob_n, np_random):
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


# our environment
class Env:
    def __init__(self):
        # Env related
        self.num_rows = NUM_ROWS
        self.num_columns = NUM_COLUMNS
        self.num_pass_loc = 5
        self.num_dest_loc = 4
        self.num_states = (
            self.num_rows * self.num_columns * self.num_pass_loc * self.num_dest_loc
        )
        self.num_actions = 6
        self.max_row = self.num_rows - 1
        self.max_col = self.num_columns - 1
        self.locs = [(0, 0), (0, 8), (8, 0), (8, 7)]
        self.initial_states = np.zeros(self.num_states)
        self.reward_table = {
            state: {action: [] for action in range(self.num_actions)}
            for state in range(self.num_states)
        }
        # Game related
        self.taxi = Taxi()
        self.passenger = Passenger()
        self.destination = Destination()
        self.roads = pygame.sprite.Group()
        self.obstacles = pygame.sprite.Group()
        self.make_roads()
        self.make_obstacles()

        # Creating reward table
        for row in range(self.num_rows):  # 9
            for col in range(self.num_columns):  # 9
                for pass_idx in range(len(self.locs) + 1):  # 5
                    for dest_idx in range(len(self.locs)):  # 4
                        state = self.encode(row, col, pass_idx, dest_idx)
                        if pass_idx < 4 and pass_idx != dest_idx:
                            self.initial_states[state] += 1
                        for action in range(self.num_actions):
                            new_row, new_col, new_pass_idx = row, col, pass_idx
                            reward = -1
                            done = False
                            taxi_loc = (row, col)

                            # down
                            if (action == 0) and (
                                col * BOX_WIDTH,
                                (row + 1) * BOX_HEIGHT,
                            ) not in obstacleHorizontal:
                                new_row = min(row + 1, self.max_row)
                            # up
                            elif (action == 1) and (
                                col * BOX_WIDTH,
                                row * BOX_HEIGHT,
                            ) not in obstacleHorizontal:
                                new_row = max(row - 1, 0)
                            # right
                            if (action == 2) and (
                                (col + 1) * BOX_WIDTH,
                                row * BOX_HEIGHT,
                            ) not in obstacleVertical:
                                new_col = min(col + 1, self.max_col)
                            # left
                            elif (action == 3) and (
                                col * BOX_WIDTH,
                                row * BOX_HEIGHT,
                            ) not in obstacleVertical:
                                new_col = max(col - 1, 0)
                            elif action == 4:  # pickup
                                if pass_idx < 4 and taxi_loc == self.locs[pass_idx]:
                                    new_pass_idx = 4
                                else:  # passenger not at location
                                    reward = -10
                            elif action == 5:  # dropoff
                                if (taxi_loc == self.locs[dest_idx]) and pass_idx == 4:
                                    new_pass_idx = dest_idx
                                    done = True
                                    reward = 20
                                elif (taxi_loc in self.locs) and pass_idx == 4:
                                    new_pass_idx = self.locs.index(taxi_loc)
                                else:  # dropoff at wrong location
                                    reward = -10
                            new_state = self.encode(
                                new_row, new_col, new_pass_idx, dest_idx
                            )
                            self.reward_table[state][action].append(
                                (1.0, new_state, reward, done)
                            )
        self.initial_states /= self.initial_states.sum()
        self.seed()
        self.s = get_state(self.initial_states, self.np_random)

    # to help random function to not generate the same state number everytime it is called
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # reset the environment and give back a new state
    def reset(self):
        self.s = get_state(self.initial_states, self.np_random)
        return int(self.s)

    # take an action and return the reward table for that action
    def step(self, a):
        transitions = self.reward_table[self.s][a]
        # i = get_state([t[0] for t in transitions], self.np_random)
        # print("I STEP: ", i)
        # print(transitions)
        # p, s, r, d = transitions[i]
        p, s, r, d = transitions[0]
        self.s = s
        return (int(s), r, d, {"prob": p})

    # randomly get an action from the action space
    def get_action(self):
        return self.np_random.randint(self.num_actions)

    # encode current row, col, pass_index, dest_index and return a state for it
    def encode(self, taxi_row, taxi_col, pass_index, dest_index):
        i = taxi_row
        i *= self.num_columns
        i += taxi_col
        i *= self.num_pass_loc
        i += pass_index
        i *= self.num_dest_loc
        i += dest_index
        return i

    # decode the state and return row, col, pass_index, dest_index
    def decode(self, i):
        out = []
        out.append(i % self.num_dest_loc)
        i = i // self.num_dest_loc
        out.append(i % self.num_pass_loc)
        i = i // self.num_pass_loc
        out.append(i % self.num_columns)
        i = i // self.num_rows
        out.append(i)
        return reversed(out)

    def make_roads(self):
        x, y = 0, 0
        for i in range(self.num_rows):
            for j in range(self.num_columns):
                new_road = Road()
                new_road.rect.x = x
                new_road.rect.y = y
                self.roads.add(new_road)
                x += BOX_WIDTH
            x = 0
            y += BOX_HEIGHT

    def make_obstacles(self):
        for coord in obstacleVertical:
            new_obstacle = Obstacle(width=6, height=BOX_HEIGHT)
            new_obstacle.rect.x = coord[0] - (new_obstacle.surf.get_width() / 2)
            new_obstacle.rect.y = coord[1]
            self.obstacles.add(new_obstacle)

        for coord in obstacleHorizontal:
            new_obstacle = Obstacle(width=BOX_WIDTH, height=6)
            new_obstacle.rect.x = coord[0]
            new_obstacle.rect.y = coord[1] - (new_obstacle.surf.get_height() / 2)
            self.obstacles.add(new_obstacle)

    def alert_obstacles(self, taxi_coord_row, taxi_coord_col):
        if (
            taxi_coord_col * BOX_WIDTH,
            taxi_coord_row * BOX_HEIGHT,
        ) in obstacleVertical:
            print(bcolors.WARNING + "OBSTACLE AT LEFT" + bcolors.ENDC)
        if (
            (taxi_coord_col + 1) * BOX_WIDTH,
            taxi_coord_row * BOX_WIDTH,
        ) in obstacleVertical:
            print(bcolors.WARNING + "OBSTACLE AT RIGHT" + bcolors.ENDC)
        if (
            taxi_coord_col * BOX_WIDTH,
            taxi_coord_row * BOX_HEIGHT,
        ) in obstacleHorizontal:
            print(bcolors.WARNING + "OBSTACLE AT UP" + bcolors.ENDC)
        if (
            taxi_coord_col * BOX_WIDTH,
            (taxi_coord_row + 1) * BOX_HEIGHT,
        ) in obstacleHorizontal:
            print(bcolors.WARNING + "OBSTACLE AT DOWN" + bcolors.ENDC)

    def render_roads(self):
        for road in self.roads:
            window.blit(road.surf, road.rect)
        # for column in range(0, WINDOW_WIDTH, BOX_WIDTH):
        #     pygame.draw.line(window, WHITE, (column, 0), (column, WINDOW_HEIGHT))
        # for row in range(0, WINDOW_HEIGHT, BOX_HEIGHT):
        #     pygame.draw.line(window, WHITE, (0, row), (WINDOW_WIDTH, row))

    def render_obstacles(self):
        for obstacle in self.obstacles:
            window.blit(obstacle.surf, obstacle.rect)

    def render(self, fps):
        # render roads
        self.render_roads()

        # render obstacles
        self.render_obstacles()

        # decode the current state and get the taxi's row and col
        # as well as pass_index and dest_index
        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)

        # if passenger is not in taxi
        # use the pass_index to get the passenger coordinate
        # and then set the passenger's x, y
        if pass_idx < 4:
            self.passenger.rect.x = (
                passDesCoord[pass_idx][0] + BOX_WIDTH / 2
            ) - self.passenger.surf.get_width() / 2
            self.passenger.rect.y = (
                passDesCoord[pass_idx][1] + BOX_HEIGHT / 2
            ) - self.passenger.surf.get_height() / 2
            # draw the passenger
            window.blit(self.passenger.surf, self.passenger.rect)

        if (taxi_col * BOX_WIDTH, taxi_row * BOX_HEIGHT) == passDesCoord[dest_idx]:
            self.passenger.rect.x = (
                passDesCoord[dest_idx][0] + BOX_WIDTH / 2
            ) - self.passenger.surf.get_width() / 2
            self.passenger.rect.y = (
                passDesCoord[dest_idx][1] + BOX_HEIGHT / 2
            ) - self.passenger.surf.get_height() / 2
            window.blit(self.passenger.surf, self.passenger.rect)

        # use the taxi_row and taxi_col to get the coordinate
        # and then set the taxi's x,y
        self.taxi.rect.x = taxi_coord[taxi_row][taxi_col][0]
        self.taxi.rect.y = taxi_coord[taxi_row][taxi_col][1]

        # set destinaton's x,y coordinate for drawing
        self.destination.rect.x = (
            passDesCoord[dest_idx][0] + BOX_WIDTH / 2
        ) - self.passenger.surf.get_width() / 2

        self.destination.rect.y = (
            passDesCoord[dest_idx][1] + BOX_HEIGHT / 2
        ) - self.passenger.surf.get_height() / 2

        # draw the destination and taxi
        window.blit(self.destination.surf, self.destination.rect)
        window.blit(self.taxi.surf, self.taxi.rect)

        self.alert_obstacles(taxi_row, taxi_col)

        # update the window
        pygame.display.flip()

        # fps
        clock.tick(fps)
