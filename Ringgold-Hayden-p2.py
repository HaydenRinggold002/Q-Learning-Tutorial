# -*- coding: utf-8 -*-
"""
# Starter code for project 2. 
"""

import numpy as np
import random

class RacecarMDP:
    """
    # This class defines the race MDP.
    # Your agent will need to interact with an instantiation of this class as part of your implementation of Q-learning.
    # Note that your agent should not access anything from this class except through the apply_action() function. 
    """
    
    def __init__(self):
        # states
        self.states = ["Cool", "Warm", "Overheated"]

        # terminal_states
        self.is_terminal = {"Cool": False,
                            "Warm": False,
                            "Overheated": True}

        # start state
        self.current_state = "Cool"

        # actions
        self.actions = ["Slow", "Fast"]

        # transition model. P(s' | s, a)
        self.transition_model = {
            ("Cool", "Slow"): {"Cool": 1.0, "Warm": 0.0, "Overheated": 0.0}, # P(s' | Cool, Slow)
            ("Cool", "Fast"): {"Cool": 0.5, "Warm": 0.5, "Overheated": 0.0}, # P(s' | Cool, Fast)
            ("Warm", "Slow"): {"Cool": 0.5, "Warm": 0.5, "Overheated": 0.0}, # P(s' | Warm, Slow)
            ("Warm", "Fast"): {"Cool": 0.0, "Warm": 0.0, "Overheated": 1.0}  # P(s' | Warm, Fast)
        }

    
    def get_reward(self, action):
        # Defines the reward function.
        if self.current_state != "Overheated":
            if action == "Fast":
                return 20
            elif action == "Slow":
                return 10

        else:
            # Current state is Overheated
            return -50


    def apply_action(self, action):
        """
        This function updates the environment state in response to some action taken.
        It returns the new state and the reward received. 
        """
        if self.current_state == "Overheated":
            print("your racecar has overheated :(")
            return 0, 0

        else:

            # Randomly select successor state according to transition probabilities
            successor_state = np.random.choice(self.states,
                                                        p=[self.transition_model[(self.current_state, action)][sucessor] for sucessor in self.states])

            # Update the state
            self.current_state = successor_state

            # Reward
            reward = self.get_reward(action)

            return reward


    def reset_mdp(self):
        self.current_state = "Cool"
        

class Agent(RacecarMDP):
    
    def __init__(self, r_plus, n_e, gamma=1.0):
        """
        # Call this function to instantiate a Q-learning agent. Examples:
        # >> my_agent = Agent(100, 5)
        # >> my_agent_w_discounting = Agent(100, 5, gamma=0.8)
        #
        # For this project, it is fine to just leave gamma on 1.0 (but you should explain what that means
        # in the tutorial.
        #
        """
        
        mdp = RacecarMDP()
        self.r_plus = r_plus  # Optimistic reward value (see exploration function) 
        self.n_e = n_e        # Count threshold (see exploration function)
        self.gamma = gamma    # Discount factor

        # ----------------------------------------------------------------------------------------#
        # Initialize the following. For the tables, there are lots of ways you might implement    #
        # them, but I would consider using dictionaries or 2-dimensional lists.                   #
        # It is safe to assume that the agent knows the full space of states and actions that are # 
        # possible.                                                                               #
        # For gamma, r_plus, and n_e: you can try any values you like and see how it goes.        #
        # ----------------------------------------------------------------------------------------#
        # initialize the q and n tables to all 0s
        num_states = len(mdp.states)
        num_actions = len(mdp.actions)
        # creates an array of shape num_states by num_actions
        self.q_table = np.zeros((num_states,num_actions))      # Note that this is referred to as Q in the algorithm
        self.n_table = np.zeros((num_states, num_actions))      # Note that this is referred to as N_sa in the algorithm


    def do_q_learning(self, max_trials=100, max_trans=100):
        """
        # Outer function for Q-learning. The main Q-learning algorithm will need to be implemented in
        # the function update_and_choose_action(). You will also need to implement 
        """
        
        mdp = RacecarMDP()
        trial_num = 1

        while trial_num <= max_trials:

            # Set up the MDP and initialize the reward to 0
            mdp.reset_mdp()
            r = 0
            s_prev = None

            # Main Q-learning loop:
            trans_num = 0
            
            while trans_num <= max_trans:

                # Check to see if the environment is in a terminal state. If so, we have to end the trial and start a new one.
                if mdp.current_state == 'Overheated':
                    break
                
                # set the current state to an integer to traverse n and q tables (for exploitation vs exploration)
                if(mdp.current_state == 'Cool'):
                    cs = 0
                elif(mdp.current_state == 'Warm'):
                    cs = 1
                else:
                    cs = 2
                # set the current action to an integer to traverse n table
                if(next_action == 'Slow'):
                    ca = 0
                elif(next_action == 'Fast'):
                    ca = 1
                # Update the agent and get the next action.
                next_action = self.update_and_choose_action(s_prev, mdp.current_state, r, self.gamma, ca, cs)
                
                # set the current state to an integer to traverse n table
                if(mdp.current_state == 'Cool'):
                    cs = 0
                elif(mdp.current_state == 'Warm'):
                    cs = 1
                else:
                    cs = 2
                # set the current action to an integer to traverse n table
                if(next_action == 'Slow'):
                    ca = 0
                elif(next_action == 'Fast'):
                    ca = 1
                # update the n table based on the action
                self.n_table[cs,ca] = self.n_table[cs,ca]+1

                # Apply the action to the environment and get the resulting reward.
                r = mdp.apply_action(next_action)

                # Increment the transition counter
                trans_num += 1

                # -------------------------------------------------------------------------------------------------#
                # This would probbaly be a good place to print stuff out for the transition that has now been made:
                # -------------------------------------------------------------------------------------------------#
                #
                print("The current state is: " + mdp.current_state)
                print("The selected action was: " + next_action)
                print("The reward was " + r)
                #
                #
                #

            # Increment the trial counter
            trial_num += 1

        # -------------------------------------------------------------------------------------------------#
        # After everything is finished, be sure to display the resulting policy
        # which can be derived from the Q table.
        # -------------------------------------------------------------------------------------------------#
        self.print_policy()  # You will need to implement this function below.
        

    def update_and_choose_action(self, s_prev, s_curr, r, gamma, cs, ca):
        """
        # This function should be a translation of the Q-learning algorithm from the project document.
        # It should take the current state and current reward as input (the percept) and output the next
        # action the agent should take.
        #
        # s_curr: the current state. s' in the algorithm.
        # r     : the transition reward just received
        # gamma : the discount factor
        """
        
        if s_prev is not None:
            # first call f to see if we are exploring or exploiting
            exp = self.f(self, self.q_table[cs,ca], self.n_table[cs,ca])
            if exp:
                # if we do end up with exploration we just want to randomly choose an action
                return np.random.choice(['Slow', 'Fast'])
            else:
                na = np.argmax(self.q_table[cs])
                if(na == self.q_table[0,0]):
                    return 'Slow'
                elif(na == self.q_table[0,1]):
                    return 'Fast'
                elif(na == self.q_table[1,0]):
                    return 'Slow'
                elif(na == self.q_table[1,1]):
                    return 'Fast'
            # ----------------------------------------------------------------------------------------#
            # Implement what should normally happen here using the Q-Learning-Agent algorithm:        #
            # ----------------------------------------------------------------------------------------#
            #
            # 
            #
            # 
            
        else:
            
            # Otherwise, if s_curr is the very first state in a trial (i.e., there have not been any transitions yet),
            # then the best we can do is choose an action at random:
            return np.random.choice(['Slow', 'Fast']) 
        


    def f(self, u, n):
        """
        # Exploration function that you will need to implement.
        #
        # u: the utility value
        # n: the number of times the state-action pair has been tried
        """

        # ----------------------------------------------------------------------------------------#
        # Implement the exploration function here:                                                #
        # ----------------------------------------------------------------------------------------#
        # 
        x = random.randint(0,50)
        x = x/100
        # this will ensure that at the beginning u/n will be large but
        # as the number of loops increases (number of times we've been in this state)
        # exploration will happen less frequently
        if(x > u/(n+1)):
            return 1
        else:
            return 0
        



    def print_policy(self):
        """
        # Function that uses self.q_table to print the action each agent should take from each non-terminal state. 
        """
        # ----------------------------------------------------------------------------------------#
        # Implement the function here:                                                            #
        # ----------------------------------------------------------------------------------------#
        #
        if(self.q_table[0,0] > self.q_table[0,1]):
            print("The optimal action for the cold state is slow")
        else:
            print("The optimal action for the cold state is fast")
            
        if(self.q_table[1,0] > self.q_table[1,1]):
            print("The optimal action for the warm state is slow")
        else:
            print("The optimal action for the warm state is fast")
            
        #if(self.q_table[2,0] < 0 or self.q_table[2,1] < 0):
        #    print("You do not want to enter the terminal state overheated")

        
        
        


# Example usage for how to run your agent:
my_agent = Agent(10, 100)
my_agent.__init__
my_agent.do_q_learning
my_agent.print_policy
print(my_agent.q_table)
print(my_agent.n_table)
# my_agent.do_q_learning()

