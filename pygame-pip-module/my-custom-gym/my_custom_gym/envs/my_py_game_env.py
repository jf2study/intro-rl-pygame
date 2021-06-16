#!pip install opencv-python
#!pip install pillow
#!pip install pygame
from io import StringIO
import numpy as np
import pygame
import cv2
import time
import PIL.Image
import random
import PIL.Image as Image
import gym
from gym import Env, spaces

class Player(pygame.sprite.Sprite):
    """
    Spawn a player
    """

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.movex = np.random.choice(np.arange(40,900))
        self.movey = 0
        self.frame = 0
        self.health = 10
        self.images = []
        self.limits = [0, 930]
        self.ALPHA = (0, 255, 0)
        self.ani = 4

        for i in range(1, 5):

            img = pygame.image.load('hero' + str(i) + '.png').convert()
            img.convert_alpha()
            img.set_colorkey(self.ALPHA)
            self.images.append(img)
            self.image = self.images[0]
            self.rect = self.image.get_rect()

    def control(self, x, y):
        """
        control player movement
        """

        # self.movex += x
        self.movex = x * 10
        self.movey += y

    def update(self):
        """
        Update sprite position
        """

        self.rect.x = self.rect.x + self.movex
        self.rect.y = self.rect.y + self.movey
        # print(f'self move x,y {self.rect.x},{self.movey}')
        reward = 0

        # moving left
        if self.movex < 0:
            self.frame += 1
            if self.frame > 3 * self.ani:
                self.frame = 0
            self.image = pygame.transform.flip(self.images[self.frame // self.ani], True, False)

        # moving right
        if self.movex > 0:
            self.frame += 1
            if self.frame > 3 * self.ani:
                self.frame = 0
            self.image = self.images[self.frame // self.ani]

        self.hit_list = pygame.sprite.spritecollide(self, self.enemy_list, False)
        if not self.check_collisions():
            for e in self.enemy_list:
                distance = abs(self.rect.x - e.rect.x)
                if distance < 150 and distance > 100:
                    reward = 25
                    # print(f'Reward {self.health}')
        else:
            reward = -100
        return reward


    def check_collisions(self):
        """
        Check collisions sprite position
        """
        reward = 0
        is_hit = False
        for enemy in self.hit_list:
            is_hit = True
            # self.health -= 5
            # reward = -5
            # print(f'Health {self.health} {enemy.rect.x}')
        return is_hit
        # return reward


class Enemy(pygame.sprite.Sprite):
    """
    Spawn an enemy
    """

    def __init__(self, x, y, img):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.image.load(img)
        self.image.convert_alpha()
        self.ALPHA = (0, 255, 0)
        self.image.set_colorkey(self.ALPHA)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.counter = 0

    def move(self):
        """
        enemy movement
        """
        distance = 80
        speed = 8

        if self.counter >= 0 and self.counter <= distance:
            self.rect.x += speed
        elif self.counter >= distance and self.counter <= distance * 2:
            self.rect.x -= speed
        else:
            self.counter = 0

        self.counter += 1


class Level():
    def bad(lvl, eloc):
        if lvl == 1:
            enemy = Enemy(eloc[0], eloc[1], 'enemy.png')
            enemy_list = pygame.sprite.Group()
            enemy_list.add(enemy)
        if lvl == 2:
            print("Level " + str(lvl))

        return enemy_list


class MyPyGameEnv(gym.Env):

    def __init__(self):
        self.done = False

        # https://github.com/nicknochnack/OpenAI-Reinforcement-Learning-with-Custom-Environment/blob/main/OpenAI Custom Environment Reinforcement Learning.ipynb
        # https://www.youtube.com/watch?v=bD6V3rcr_54

        ######################################

        self.waiting = False
        self.timer_event = pygame.USEREVENT + 1
        self.num_envs = 1
        self.worldx = 960
        self.worldy = 180  # 360 #720
        self.fps = 40
        self.ani = 4
        self.ALPHA = (0, 255, 0)
        self.world = pygame.display.set_mode([self.worldx, self.worldy])
        self.counter = 0


        self.view = pygame.surfarray.array3d(self.world)

        self.player = Player()
        self.player_list = pygame.sprite.Group()
        self._episode_ended = False

        self._state = 0

        self.player.ALPHA = self.ALPHA

        self.backdrop = pygame.image.load('stage.jpg')
        self.clock = pygame.time.Clock()
        self.timer_event = pygame.USEREVENT + 1

        self.backdropbox = self.world.get_rect()

        self.eloc = [np.random.choice(np.arange(100, 900)), 0]
        self.ep_return = 10

        pygame.init()
        self.TIME_LIMIT = 10
        self.start_ticks = pygame.time.get_ticks()
        pygame.time.set_timer(self.timer_event, 1000)
        self.main = True
        # Define a 2-D observation space
        # Define an action space ranging from 0 to 4
        self._action_space = spaces.Discrete(3, )

        # OPEN AI

        self.observation_space = spaces.Box(low=0, high=255, shape=(self.worldy, self.worldx, 3), dtype=np.uint8)

        self.action_space = self._action_space

        self.enemy_list = Level.bad(1, self.eloc)
        self.player.enemy_list = self.enemy_list
        self._current_time_step = self.step(0)


    def draw_elements_on_canvas(self):
        ###################################

        font = pygame.font.SysFont("comicsansms", 30)

        text = font.render(' Health {0:05d} '.format(self.player.health), True, (80, 128, 0))

        # https://nerdparadise.com/programming/pygame/part5
        # https://www.pygame.org/docs/tut/tom_games2.html
        self.backdrop.fill((255, 255, 255),
                           rect=text.get_rect(topleft=(80 - text.get_width() // 2, 120 - text.get_height() // 2)))

        self.backdrop.blit(text,
                           (80 - text.get_width() // 2, 120 - text.get_height() // 2))

        self.world.blit(self.backdrop, self.backdropbox)
        new_reward = 0

        self.player_list.draw(self.world)
        self.enemy_list.draw(self.world)
        for e in self.enemy_list:
            e.move()
        pygame.display.flip()
        new_reward = self.player.update()
        return new_reward
        ###################################
        # Init the canvas

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def reset(self):

        self.counter += 1
        self.done = False
        pygame.init()
        self.start_ticks = pygame.time.get_ticks()
        if self.counter % 100 == 0:
            print(f'time {self.start_ticks}')

        self.eloc = []
        self.eloc = [300, 0]
        # Initialize the elements
        self.enemy_list = Level.bad(1, self.eloc)

        # Reset the reward
        self.ep_return = 10
        self.player = Player()  # spawn player
        self.player.rect.x = 0  # go to x
        self.player.rect.y = 30  # go to y
        self.player_list = pygame.sprite.Group()
        self.player_list.add(self.player)
        self.steps = 10

        self.player.enemy_list = self.enemy_list

        # Draw elements on the canvas
        # world = pygame.surfarray.make_surface(self.view)
        self.world.blit(self.backdrop, self.backdropbox)
        self.player.update()
        self.player_list.draw(self.world)
        self.enemy_list.draw(self.world)
        for e in self.enemy_list:
            e.move()
        pygame.display.flip()
        self.view = pygame.surfarray.array3d(self.world)
        reward = 0
        self._episode_ended = False


        return self.view.transpose([1, 0, 2])


    # This is for open ai custom env
    def close(self):
        pass


    def step(self, action):


        ## Make sure episodes don't go on forever.

        # Flag that marks the termination of an episode
        self.done = False

        # Assert that it is a valid action
        # assert self._action_space.contains(action), "Invalid Action"

        # Reward for executing a step.
        reward = 0

        # apply the action to the chopper
        if action == -1:
            self.player.control(0, 0)

        elif action == 0:
            if self.player.rect.x > 40:
                self.player.control(-1, 0)
                reward = 10

        elif action == 1:
            if self.player.rect.x < 900:
                self.player.control(1, 0)
                reward = 10
            else:
                self.player.control(0, 0)

        # self._current_time_step = self.step(action)
        seconds = (pygame.time.get_ticks() - self.start_ticks) / 1000
        if seconds >= self.TIME_LIMIT:
            self.done = True
        # if self.fuel_left == 0:
        #    done = True

        # Draw elements on the canvas
        reward += self.draw_elements_on_canvas()
        self.player.health += reward
        self._state = self.player.health
        self.ep_return += reward
        # print("New reward ",reward)
        pygame.display.flip()
        self.view = pygame.surfarray.array3d(self.world)


        if self.done:

            self._episode_ended = True
            return self.view.transpose([1, 0, 2]), reward, self.done, {}
        else:
            return self.view.transpose([1, 0, 2]), reward, self.done, {}


    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def render(self, mode="human"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""

        self.draw_elements_on_canvas()

        pygame.display.flip()
        self.view = pygame.surfarray.array3d(self.world)

        #  convert from (width, height, channel) to (height, width, channel)

        self.clock.tick(self.fps)

        if mode == "human":

            PIL.Image.fromarray(self.view.transpose([1, 0, 2]))

        elif mode == "rgb_array":

            return self.view.transpose([1, 0, 2])
