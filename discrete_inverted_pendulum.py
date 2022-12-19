# simulation
import pygame, sys

# mathematical arrays, matrices, and calculations
import numpy as np

# pygame constants
from pygame.locals import *

# math library
import math

# Reinforcement Learning Controller
import RL_controller

# Parser for command line options, arguements, and sub-commands
import argparse

# time-related functions
import time

# interfacing cameras to capture images 
import pygame.camera

# Image processing capabilities for python interpreter
from PIL import Image

# MATLAB for changing figures
import matplotlib.pyplot as plt


# [-90, 90] x 20 -> 4.5
# p-5, 5] x 20

# Class 1: Discrete Inverted Pendulum Object = 9 Functions
class DiscreteInvertedPendulum(object):

    # Function 1.1: Initialize the discrete inverted pendulum
    def __init__(self,args,windowdims,cartdims,penddims,action_range=[-1,1]):
        
        # variable to store the arguements
        self.args = args
        
        # variable to store the action range
        self.action_range = action_range

        # variable to store the width of the window
        self.window_width = windowdims[0]
        
        # variable to store the height of the window
        self.window_height = windowdims[1]

        # variable to store the width of the cart
        self.cart_width = cartdims[0]
        
        # variable to store the height of the cart
        self.cart_height = cartdims[1]
        
        # variable to store the width of the pendulum
        self.pendulum_width = penddims[0]
        
        # variable to store the length of the pendulum
        self.pendulum_length = penddims[1]

        # variable to store the cart height
        self.Y_CART = 3 * self.window_height / 4
        
        # reset the discrete pendulum object
        self.reset()
        
        # if noise will added to gravity and mass
        if args.add_noise_to_gravity_and_mass:
            
            # set the gravity of the discrete pendulum object
            self.gravity = args.gravity + np.random.uniform(-5, 5)
            
            # set the mass of the cart
            self.masscart = 1.0 + np.random.uniform(-0.5, 0.5)
            
            # set the mass of the pole
            self.masspole = 0.1 + np.random.uniform(-0.05, 0.2)
        
        # if no noise will be added to the gravity and mass
        else:
            
            # set the gravity
            self.gravity = args.gravity
            
            # set the mass of the cart
            self.masscart = 1.0
            
            # set the mass of the pole
            self.masspole = 0.1
        
        # compute the total mass
        self.total_mass = self.masspole + self.masscart
        
        # set the length (half of the pole's length)
        self.length = 0.5
        
        # set the pole mass length
        self.polemass_length = self.masspole * self.length
        
        # set the force magnitude
        self.force_mag = 10.0
        
        # set the delta time ~ seconds between state updates
        self.dt = args.dt
        
        # theta threshold in radians ~ angle at which to fail the episode
        self.theta_threshold_radians =  math.pi / 2
        
        # position threshold
        self.x_threshold = 2.4
        
        # angular velocity threshold
        self.theta_dot_threshold = 7

        # conversion of position
        self.x_conversion = self.window_width / 2 / self.x_threshold

    # Function 1.2: Reset function ~ initializes pendulum in upright state with small perturbation
    def reset(self):
        
        # set terminal to false
        self.terminal = False
        
        # set time step
        self.timestep = 0

        # set the velocity
        self.x_dot = np.random.uniform(-0.03, 0.03)
        
        # set the position
        self.x = np.random.uniform(-0.01, 0.01)

        # set the theta angle
        self.theta = np.random.uniform(-0.03, 0.03)
        
        # set the angular velocity
        self.theta_dot = np.random.uniform(-0.01, 0.01)
        
        # set the total reward 
        self.total_reward = 0
        
        # set the reward
        self.reward = 0

    # Function 1.3: Function to get the reward
    def get_reward(self, discrete_theta):
        
        # small survival reward
        smallSurvivalReward = 0.05

        # angle reward
        angleReward = 0.95 * (self.theta_threshold_radians-np.abs(self.from_discrete(discrete_theta, self.args.theta_discrete_steps,range=[-math.pi/2, math.pi/2])))/self.theta_threshold_radians

        # return the reward
        return smallSurvivalReward+angleReward

    # Function 1.4: function to discretize ~ represent using a discrete quantity
    def to_discrete(self,value,steps,range):
        
        # limit the values and store them in a variable ~ threshold
        value = np.clip(value,range[0],range[1])
        
        # normalize to [0,1]
        value = (value-range[0])/(range[1]-range[0])
        
        # Ensure it cannot be exactly steps
        value = int(value*steps*0.99999)
        
        # return the value
        return value

    # Function 1.5: function to connect
    def from_discrete(self,discrete_value,steps,range):
        
        # on average the discrete value gets rounded down even if it was 19.99 -> 19 so we use +0.5 as more accurate
        value = (discrete_value+0.5)/steps 

        # update the value
        value = value*(range[1]-range[0])+range[0]
        
        # return the value
        return value

    # Function 1.6: function to get the continuous values
    def get_continuous_values(self):
        
        # return the continuous values
        return (self.terminal,self.timestep,self.x,self.x_dot,self.theta,self.theta_dot,self.reward)

    # Function 1.7: function to get the discrete values
    def get_discrete_values(self):
        
        # compute discrete theta
        discrete_theta = self.to_discrete(self.theta,self.args.theta_discrete_steps,range=[-math.pi/2,math.pi/2])
        
        # compute discrete angular velocity
        discrete_theta_dot = self.to_discrete(self.theta_dot,self.args.theta_dot_discrete_steps,range=[-self.theta_dot_threshold,self.theta_dot_threshold])

        # return the discrete values
        return (self.terminal,self.timestep,discrete_theta,discrete_theta_dot,self.reward)

    # Function 1.8: function to set the state
    def set_state(self,state):
        
        # seperate the state values into different variables
        terminal,timestep,x,x_dot,theta,theta_dot = state
        
        # set the terminal
        self.terminal = terminal
        
        # set the time step
        self.timestep = timestep
        
        # set the position
        self.x = x
        
        # set the velocity
        self.x_dot = x_dot
        
        # set the theta angle in radians
        self.theta = theta
        
        # set the angular velocity in radians
        self.theta_dot = theta_dot

    # Function 1.9: function to apply force on the cart
    def step(self, action):
        
        # move left
        if action == 0:
            
            # apply a force of -1 on the cart
            force = -10
        
        # stay still
        elif action == 1:
            
            # apply a force of 0 on the cart
            force = 0

        # move right
        elif action == 2:
            
            # apply a force of 10 on the cart
            force = 10
        
        # if other number recieved
        else:
            
            # error
            raise Exception("Invalid Action, Actions are only 0, 1, 2")
        
        # increment the time step
        self.timestep += 1

        # set the cosine of the theta
        costheta = math.cos(self.theta)

        # set the sin of the theta
        sintheta = math.sin(self.theta)
        
        # temp value
        temp = (force + self.polemass_length * self.theta_dot ** 2 * sintheta) / self.total_mass

        # angular acceleration
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))

        # point acceleration
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # position
        self.x = self.x + self.dt * self.x_dot
        
        # velocity
        self.x_dot = self.x_dot + self.dt * xacc
        
        # theta
        self.theta = self.theta + self.dt * self.theta_dot
        
        # angular velocity
        self.theta_dot = self.theta_dot + self.dt * thetaacc

        # terminal
        self.terminal = bool(self.x < -self.x_threshold or self.x > self.x_threshold or self.theta < -self.theta_threshold_radians or self.theta > self.theta_threshold_radians)
        
        # radians to degrees
        # within -+ 15
        # if (self.theta * 57.2958) < 15 and (self.theta * 57.2958) > -15:
        #     self.score += 1
        #     self.reward = 1
        # else:
        #     self.reward = 0

        # reward
        self.reward = self.get_reward(self.to_discrete(self.theta, self.args.theta_discrete_steps, range=[-math.pi/2, math.pi/2]))
        
        # total reward
        self.total_reward = self.total_reward + self.reward
        
        # return the continuous values
        return self.get_continuous_values()


# Class 2: Inverted pendulum game = 10 Functions
class InvertedPendulumGame(object):

    # Function 2.1: function to initialize the inverted pendulum game
    def __init__(self, args, windowdims=(800, 400), cartdims=(50, 10), penddims=(6.0, 150.0), refreshfreq=1000, mode=None):

        #
        self.args = args
        
        #
        self.RL_controller = mode
        
        #
        self.max_timestep = args.max_timestep
        
        #
        self.game_round_number = 0
        
        #
        self.pendulum = DiscreteInvertedPendulum(args, windowdims, cartdims, penddims)
        
        #
        self.performance_figure_path = args.performance_figure_path

        #
        self.window_width = windowdims[0]
        
        #
        self.window_height = windowdims[1]

        #
        self.cart_width = cartdims[0]
        
        #
        self.cart_height = cartdims[1]
        
        #
        self.pendulum_width = penddims[0]
        
        #
        self.pendulum_length = penddims[1]
        
        #
        self.manual_action_magnitude = args.manual_action_magnitude
        
        #
        self.random_controller = args.random_controller
        
        #
        self.noisy_actions = args.noisy_actions

        #
        self.score_list = []

        #
        self.Y_CART = self.pendulum.Y_CART
        
        # self.time gives time in frames
        self.timestep = 0

        #
        pygame.init()
        
        #
        self.clock = pygame.time.Clock()
        
        # specify number of frames / state updates per second
        self.REFRESHFREQ = refreshfreq
        
        #
        self.surface = pygame.display.set_mode(windowdims, 0, 32)
        
        #
        pygame.display.set_caption('Inverted Pendulum Game')
        
        # array specifying corners of pendulum to be drawn
        self.static_pendulum_array = np.array([[-self.pendulum_width / 2, 0],[self.pendulum_width / 2, 0],[self.pendulum_width / 2, -self.pendulum_length],[-self.pendulum_width / 2, -self.pendulum_length]]).T
        
        #
        self.BLACK = (0, 0, 0)
        
        #
        self.RED = (255, 0, 0)
        
        #
        self.WHITE = (255, 255, 255)

    # Function 2.2
    def draw_cart(self, x, theta):
        
        #
        cart = pygame.Rect(self.pendulum.x * self.pendulum.x_conversion + self.pendulum.window_width / 2 - self.cart_width // 2,self.Y_CART, self.cart_width, self.cart_height)
        
        #
        pygame.draw.rect(self.surface, self.RED, cart)
        
        #
        pendulum_array = np.dot(self.rotation_matrix(-theta), self.static_pendulum_array)
        
        #
        pendulum_array += np.array([[x * self.pendulum.x_conversion + self.pendulum.window_width / 2], [self.Y_CART]])
        
        #
        pendulum = pygame.draw.polygon(self.surface, self.BLACK,((pendulum_array[0, 0], pendulum_array[1, 0]),(pendulum_array[0, 1], pendulum_array[1, 1]),(pendulum_array[0, 2], pendulum_array[1, 2]),(pendulum_array[0, 3], pendulum_array[1, 3])))

    # Function 2.3
    @staticmethod
    def rotation_matrix(theta):
        
        #
        return np.array([[np.cos(theta),np.sin(theta)],[-1 * np.sin(theta),np.cos(theta)]])

    # Function 2.4
    def render_text(self, text, point, position="center", fontsize=48):
        
        #
        font = pygame.font.SysFont(None, fontsize)
        
        #
        text_render = font.render(text, True, self.BLACK, self.WHITE)
        
        #
        text_rect = text_render.get_rect()
        
        #
        if position == "center":
            
            #
            text_rect.center = point
        
        #
        elif position == "topleft":
            
            #
            text_rect.topleft = point
        
        #
        self.surface.blit(text_render, text_rect)
    
    # Function 2.5
    def time_seconds(self):
        
        #
        return self.timestep / float(self.REFRESHFREQ)

    # Function 2.6
    def starting_page(self):
        
        #
        self.surface.fill(self.WHITE)
        
        #
        self.render_text("Inverted Pendulum",(0.5 * self.window_width, 0.4 * self.window_height))

        #
        self.render_text("COMP 417 Assignment 2",(0.5 * self.window_width, 0.5 * self.window_height),fontsize=30)
        
        #
        self.render_text("Press Enter to Begin",(0.5 * self.window_width, 0.7 * self.window_height),fontsize=30)

        #
        pygame.display.update()

    # Function 2.7
    def save_current_state_as_image(self, path):
        
        #
        im = Image.fromarray(self.surface_array)
        
        #
        im.save(path + "current_state.png")

    # Function 2.8
    def game_round(self):
        
        #
        LEFT = 0
        
        #
        NO_ACTION = 1
        
        #
        RIGHT = 2
        
        #
        self.pendulum.reset()
        
        #
        if not self.RL_controller is None:
            
            # reset the reinforcement learning controller
            self.RL_controller.reset()

        #
        theta_diff_list = []

        #
        action = NO_ACTION
        
        #
        for i in range(self.max_timestep):
            
            #
            self.surface_array = pygame.surfarray.array3d(self.surface)
            
            #
            self.surface_array = np.transpose(self.surface_array, [1,0,2])
            
            #
            if self.RL_controller is None:
                
                #
                for event in pygame.event.get():
                    
                    #
                    if event.type == QUIT:
                        
                        #
                        pygame.quit()
                        
                        #
                        sys.exit()
                    
                    #
                    if event.type == KEYDOWN:
                        
                        #
                        if event.key == K_LEFT:
                            
                            #
                            action = LEFT  # "Left"
                        
                        #
                        if event.key == K_RIGHT:
                            
                            #
                            action = RIGHT
                    
                    #
                    if event.type == KEYUP:
                        
                        #
                        if event.key == K_LEFT:
                            
                            #
                            action = NO_ACTION
                        
                        #
                        if event.key == K_RIGHT:
                            
                            #
                            action = NO_ACTION
                        
                        #
                        if event.key == K_ESCAPE:
                            
                            #
                            pygame.quit()
                            
                            #
                            sys.exit()
            
            #
            else:
                
                # command line arguements
                args = get_args()

                # compute the action
                action = self.RL_controller.get_action(args.random,self.pendulum.get_discrete_values(), self.surface_array,random_controller=self.random_controller, episode=self.game_round_number)
                
                #
                for event in pygame.event.get():
                    
                    #
                    if event.type == QUIT:
                        
                        #
                        pygame.quit()
                        
                        #
                        sys.exit()
                    
                    #
                    if event.type == KEYDOWN:
                        
                        #
                        if event.key == K_ESCAPE:
                            
                            #
                            print("Exiting ... ")
                            
                            #
                            pygame.quit()
                            
                            #
                            sys.exit()

            #
            if self.noisy_actions and self.RL_controller is None:
                
                #
                action = action + np.random.uniform(-0.1, 0.1)

            #
            terminal, timestep, x, _, theta, _, _ = self.pendulum.step(action)
            
            #
            theta_diff_list.append(np.abs(theta))

            #
            self.timestep = timestep
            
            #
            self.surface.fill(self.WHITE)
            
            #
            self.draw_cart(x, theta)

            #
            time_text = "t = {}".format(int(self.pendulum.timestep))
            
            #
            self.render_text(time_text, (0.1 * self.window_width, 0.1 * self.window_height),position="topleft", fontsize=40)
            
            #
            pygame.display.update()
            
            #
            self.clock.tick(self.REFRESHFREQ)
            
            # if the terminal is reached
            if terminal:

                # break
                break

        #
        self.game_round_number += 1

        #
        if(self.game_round_number%25 == 0):
            
            #
            plt.plot(np.arange(len(theta_diff_list)), theta_diff_list)
            
            #
            text = 'Time (With discount factor of: ' + str(get_args().gamma) + ')'
            plt.xlabel(text)
            
            #
            plt.ylabel('|Theta(radians)|')
            
            #
            plt.title("|Theta| vs Time")
            
            #
            plt.savefig(self.performance_figure_path + "_run_" + str(len(self.score_list)) + ".png")
            
            #
            plt.close()

        #
        self.score_list.append(int(self.pendulum.total_reward))

    # Function 2.9
    def end_of_round(self):
        
        # fill the surface with white
        self.surface.fill(self.WHITE)
        
        # draw the cart
        self.draw_cart(self.pendulum.x, self.pendulum.theta)
        
        # render the score
        self.render_text("Score: {}".format(int(self.pendulum.total_reward)),(0.5 * self.window_width, 0.3 * self.window_height))
        
        # render the average score
        self.render_text("Average Score : {}".format(np.around(np.mean(self.score_list), 3)),(0.5 * self.window_width, 0.4 * self.window_height))
        
        # render the standard deviation of the score
        self.render_text("Standard Deviation Score : {}".format(np.around(np.std(self.score_list), 3)),(0.5 * self.window_width, 0.5 * self.window_height))
        
        # render the number of runs
        self.render_text("Runs : {}".format(len(self.score_list)),(0.5 * self.window_width, 0.6 * self.window_height))
        
        # if the Reinforcement Learning Controller is not used
        if self.RL_controller is None:

            # render the play again text
            self.render_text("(Enter to play again, ESC to exit)",(0.5 * self.window_width, 0.85 * self.window_height), fontsize=30)
        
        # update the display
        pygame.display.update()
        
        # sleep for 2.0 seconds ~ 0.5 
        time.sleep(0.25)

    # Function 2.10
    def game(self):
        
        # display the starting page
        self.starting_page()
        
        # continue until forced to break
        while True:
            
            # if manual mode is chosen
            if self.RL_controller is None:
                
                # iterate over every event
                for event in pygame.event.get():
                    
                    #
                    if event.type == QUIT:
                        
                        #
                        pygame.quit()
                        
                        #
                        sys.exit()
                    
                    #
                    if event.type == KEYDOWN:
                        
                        #
                        if event.key == K_RETURN:
                            
                            #
                            self.game_round()
                            
                            #
                            self.end_of_round()
                        
                        #
                        if event.key == K_ESCAPE:
                            
                            #
                            pygame.quit()
                            
                            #
                            sys.exit()
            #
            else:  # Use the PID controller instead, ignores input expect exit
                
                #
                self.game_round()
                
                #
                self.end_of_round()
                
                #
                self.pendulum.reset()


# Function 20: Function to retrieve the arguments from the command line
def get_args():

    # create parser variable to store the command line arguemeents
    parser = argparse.ArgumentParser()
    
    # mode
    parser.add_argument('--mode', type=str, default="RL")
    
    # random controller
    parser.add_argument('--random_controller', type=bool, default=False)
    
    # noise and gravity
    parser.add_argument('--add_noise_to_gravity_and_mass', type=bool, default=False)
    
    # maximum timestep
    parser.add_argument('--max_timestep', type=int, default=3000)
    
    # delta time ~  
    parser.add_argument('--dt', type=float, default=0.01)

    # gravity
    parser.add_argument('--gravity', type=float, default=9.81)
    
    # magnitude
    parser.add_argument('--manual_action_magnitude', type=float, default=1)
    
    # seed
    parser.add_argument('--seed', type=int, default=0)
    
    # noisy actions
    parser.add_argument('--noisy_actions', type=bool, default=False)
    
    # figure
    parser.add_argument('--performance_figure_path', type=str, default="performance_figure")

    # discrete steps for theta
    parser.add_argument('--theta_discrete_steps', type=int, default=40)
    
    # discrete steps for angular velocity
    parser.add_argument('--theta_dot_discrete_steps', type=int, default=40)

    # gamma
    parser.add_argument('--gamma', type=float, default=0.99)
    
    # learning rate
    parser.add_argument('--lr', type=float, default=0.001)

    # randomness
    parser.add_argument('--random',type=float,default=0.8)

    # store the command line arguements in a variable
    args = parser.parse_args()
    
    # return the variable
    return args

# Function 21: main function
def main():
    
    # retrieve the command line arguements and store them in a variable
    args = get_args()
    
    # set a random seed using the seed arguement
    np.random.seed(args.seed)
    
    # manual mode
    if args.mode == "manual":
        
        # create an inverted pendulum game
        inv = InvertedPendulumGame(args, mode=None)
    
    # Reinforcement Learning
    else:
        
        # create an inverted pendulum game
        inv = InvertedPendulumGame(args, mode=RL_controller.RL_controller(args))
    
    # start the game
    inv.game()

# Start of program
if __name__ == '__main__':
    
    # Call the main function
    main()