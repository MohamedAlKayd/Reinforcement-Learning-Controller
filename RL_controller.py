# Numpy library for arrays and matrices
import numpy as np

# Class 1: Reinforcement Controller
class RL_controller:
    
    # Function 1: function to initialize the reinforcement controller
    def __init__(self,args):

        # Gamma value from the command line
        self.gamma = args.gamma
        
        # Learning rate from the command line
        self.lr = args.lr
        
        # state-action values
        self.Q_value = np.zeros((args.theta_discrete_steps,args.theta_dot_discrete_steps,3))
        
        # state values
        self.V_values = np.zeros((args.theta_discrete_steps,args.theta_dot_discrete_steps))
        
        # previous action
        self.prev_a = 0
        
        # Use a previous_state = None to detect the beginning of the new round e.g. if not(self.prev_s is None): ...
        self.prev_s = None

        # random counter
        self.randomCounter = 0

    # Function 2
    def reset(self):
        
        # Reset the previous state and action
        self.prev_s = None
        self.prev_a = 0

        # Update the V values
        self.V_values = self.Q_value.max(axis=2)

        # Save text function
        np.savetxt("File",self.V_values,fmt="%.4f",delimiter=",")

    # Function 3: 
    def get_action(self,randomness,state,image_state,random_controller=False,episode=0):

        # variables from the state
        terminal, timestep, theta, theta_dot, reward = state

        # if the random controller is selected
        if random_controller:
            
            # 3 possible actions (0,1,2)
            action = np.random.randint(0,3)
        
        # if the reinforcement learning controller is used
        else:
            
            # use Q values to take the best action at each state
            action = self.Q_value[theta][theta_dot].argmax()

            # Only allow random actions at the beginning of the program
            if timestep < 90 and episode < 90 and np.random.rand()>randomness:
                
                # Compute a random action
                action = np.random.randint(0,3)

        # if there is a previous state or if the previous state is theta, angular velocity
        if not(self.prev_s is None or self.prev_s == [theta, theta_dot]):
            
            # Update the Q value
            self.Q_value[self.prev_s[0],self.prev_s[1],self.prev_a] = self.Q_value[self.prev_s[0],self.prev_s[1],self.prev_a] + \
            self.lr * ((reward+self.gamma*self.Q_value[theta,theta_dot,action]) - self.Q_value[self.prev_s[0],self.prev_s[1],self.prev_a])            
        
        # update the value of the previous state with the current theta and angular velocity
        self.prev_s = [theta,theta_dot]
        
        # update the previous action with the current action
        self.prev_a = action

        # return the action
        return action