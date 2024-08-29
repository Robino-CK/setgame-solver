import set
import pygame
import planes
import planes.gui
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import gym
import reverb

import tensorflow as tf
from tf_agents.environments import utils
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700

BLACK = (0, 0, 0)

class SetEnv(py_environment.PyEnvironment):
    def __init__(self):
        pygame.init()
        size = (WINDOW_WIDTH, WINDOW_HEIGHT)
        self.screen = planes.Display(size)
        self.screen.grab = False
    

        self.screen.image.fill(BLACK)
        
    
        self.model = set.Model()
        self.model.mode = 1
        self.model.game = set.Game(0, self.model)
        
        self.view = set.View(self.model, self.screen)
        self._action_spec = array_spec.BoundedArraySpec(
           shape=(), dtype=np.int32, minimum=0, maximum=9, name='action')
        
        
        self._observation_spec = array_spec.BoundedArraySpec(
        shape=(15,4), dtype=np.int32, minimum=0, maximum=3, name='observation')
        self._state = np.zeros((15,4), dtype=np.int32)
        
        cards = self.model.game.in_play_cards
        for i,card in enumerate(cards):
            self._state[i] = [set.colors.index(card.color), set.shapes.index(card.shape), set.shades.index(card.shade), set.numbers.index(card.number)]
        
        
        self._episode_ended = False
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        print("reset")
     #   self.model = set.Model()
      #  self.model.mode = 1
       # self.model.game = set.Game(0, self.model)
        #self._state = np.zeros((15,4), dtype=np.int32)
        
        #cards = self.model.game.in_play_cards
        #for i,card in enumerate(cards):
         #   self._state[i] = [set.colors.index(card.color), set.shapes.index(card.shape), set.shades.index(card.shade), set.numbers.index(card.number)]
     
        self._episode_ended = False
        return ts.restart(self._state)


    def _step(self, action):
        print(f"step: {action}")
        reward = 0
        done = False
        
        
        cards = self.model.game.in_play_cards
        
        cards[action].been_clicked = True
        selected_cards = []
        for card in cards:
            if card.been_clicked:
               selected_cards.append(card)
        
        
        if len(selected_cards) < 3:
            return ts.transition( observation=self._state, reward=reward, discount=1.0)
            
        print("checking set")
        if set.check_set(selected_cards[0],selected_cards[1],selected_cards[2]):
            reward = 10000
            done = True
        else:
            reward = -10
            done = False
        
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                raise SystemExit
        
        self.screen.process(events)
        self.model.update()
        
   
        self.screen.update()
        self.screen.render()
        clear_selection(self.model.game.in_play_cards)
       
        self.view.draw()
        pygame.display.flip()
        time.sleep(0.001)
    
        cards = self.model.game.in_play_cards
        self._state = np.zeros((15,4), dtype=np.int32)
        for i,card in enumerate(cards):
            self._state[i] = [set.colors.index(card.color), set.shapes.index(card.shape), set.shades.index(card.shade), set.numbers.index(card.number)]
        
        self.model.update()
        return ts.transition( observation=self._state, reward=reward, discount=1.0)

    
def random_action(cards):
    selected_cards = random.choices(cards, k=3)
    
    for card in selected_cards:
        card.been_clicked = True

def clear_selection(cards):
    for card in cards:
        card.been_clicked = False
   
def find_set(cards):
    for i in range(len(cards)):
        for j in range(i+1,len(cards)):
            for k in range(j+1,len(cards)):
                if set.check_set(cards[i],cards[j],cards[k]):
                    return [cards[i],cards[j],cards[k]]
    return None   

def best_action(cards):
    card_set = find_set(cards)
    if not card_set:
        return False
        
    for card in card_set:
        card.been_clicked = True
    return True            
def run_game():
    pygame.init()
    size = (WINDOW_WIDTH, WINDOW_HEIGHT)
    screen = planes.Display(size)
    screen.grab = False
    

    screen.image.fill(BLACK)
    model = set.Model()
    model.mode = 1
    model.game = set.Game(0, model)
    
    view = set.View(model, screen)
    
    while not model.game.check_if_won():
        
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                raise SystemExit
        
        screen.process(events)
        model.update()
        clear_selection(model.game.in_play_cards)
        
   
        screen.update()
        screen.render()

        view.draw()
        pygame.display.flip()
        time.sleep(0.001)
    
    pygame.quit()

# THE MAIN LOOP
if __name__ == "__main__":
  #  run_game()
    env = SetEnv()
#    utils.validate_py_environment(env, episodes=5)

   # while True:
        
    #    env.step(0)

    #env2 = tf_py_environment.TFPyEnvironment(env)
    #eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    fc_layer_params = (100,)
    learning_rate = 1e-3
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        env.observation_spec(),
        tensor_spec.from_spec(env.action_spec()),
        fc_layer_params=fc_layer_params)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_step_counter = tf.Variable(0)
    
    tf_agent = reinforce_agent.ReinforceAgent(
        env.time_step_spec(),
        env.action_spec(),
        actor_network=actor_net,
        optimizer=optimizer,
        normalize_returns=True,
        train_step_counter=train_step_counter)
    
    tf_agent.initialize()
    
    replay_buffer_capacity = 2000 # @param {type:"integer"}

    table_name = 'uniform_table'
    replay_buffer_signature = tensor_spec.from_spec(
      tf_agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(
      replay_buffer_signature)
    table = reverb.Table(
    table_name,
    max_size=replay_buffer_capacity,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature)

    reverb_server = reverb.Server([table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    tf_agent.collect_data_spec,
    table_name=table_name,
    sequence_length=None,
    local_server=reverb_server)

    rb_observer = reverb_utils.ReverbAddEpisodeObserver(
    replay_buffer.py_client,
    table_name,
    replay_buffer_capacity
)
    driver = py_driver.PyDriver(
    env,
    py_tf_eager_policy.PyTFEagerPolicy(
    tf_agent.collect_policy, use_tf_function=True),
    [rb_observer],
    max_episodes=200)
    initial_time_step = env.reset()
    for i in range(100):
        print(i)
        initial_time_step,_ = driver.run(initial_time_step)
    
    t = 2
  