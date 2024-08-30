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
import tf_agents
from tf_agents.trajectories import StepType
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
global steps_to_win

BLACK = (0, 0, 0)

class SetEnv(py_environment.PyEnvironment):
    def __init__(self, render=True):
        self.render = render
        if render:
            pygame.init()
            size = (WINDOW_WIDTH, WINDOW_HEIGHT)
            self.screen = planes.Display(size)
            self.screen.grab = False
    

            self.screen.image.fill(BLACK)
        
    
        self.model = set.Model()
        self.model.mode = 1
        self.model.game = set.Game(0, self.model)
        if render:
            self.view = set.View(self.model, self.screen)
        self._action_spec = array_spec.BoundedArraySpec(
           shape=(), dtype=np.int32, minimum=0, maximum=21, name='action')
        
        
        self._observation_spec = array_spec.BoundedArraySpec(
        shape=(21,5), dtype=np.int32, minimum=0, maximum=3, name='observation')
        
        self._update_state(self.model.game.in_play_cards)
        self._episode_ended = False
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode_ended = False
        self.model.game = set.Game(0, self.model)
        global steps_to_win
        steps_to_win = 0
        return ts.restart(self._state)
    
    def _update_state(self, cards):
        self._state = - np.ones((21,5), dtype=np.int32)
        
        for i,card in enumerate(cards):
            self._state[i] = [set.colors.index(card.color), set.shapes.index(card.shape), set.shades.index(card.shade), set.numbers.index(card.number), 1 if card.been_clicked else 0]

    def _update_pygame(self):
        if self.render:
        
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    raise SystemExit
            
            self.screen.process(events)
       
        self.model.update()
        if self.render:
        
    
            self.screen.update()
            self.screen.render()
            
            self.view.draw()
            pygame.display.flip()
            time.sleep(0.001)
       
    def _step(self, action):
        global steps_to_win
        steps_to_win += 1
        if self.model.game.check_if_won() or self.model.game.sets_found >= 1:
            print("game won")
            return tf_agents.trajectories.termination(observation=self._state, reward=0)
        
        cards = self.model.game.in_play_cards
        if action == 21:
            if len(cards) == 21:
                return ts.transition( observation=self._state, reward=-100000, discount=0.1)
            self.model.game.add_new_cards(3)
            
            self._update_pygame()
            self._update_state(self.model.game.in_play_cards)
            return ts.transition( observation=self._state, reward=-5000, discount=0.1)
            
            
        
        reward = 0
        discount = np.array(0.7, dtype=np.float32)
        if len(cards) - 1 < action :
            return ts.transition( observation=self._state, reward=0, discount=0.1)
            #return tf_agents.trajectories.TimeStep(step_type=StepType.FIRST, observation=self._state, reward=-100, discount=0.7)
        
        cards[action].been_clicked = True
        selected_cards = []
        for card in cards:
            if card.been_clicked:
               selected_cards.append(card)
        self._update_state(self.model.game.in_play_cards)
        reward = 0
        reward = np.array(reward, dtype=np.float32)
        available_sets = find_sets(cards)
        if len(selected_cards) == 1:
            for s in available_sets:
                if selected_cards[0] in s:
                    reward = 0
                
            return ts.transition( observation=self._state, reward=reward, discount=0.5)
            return tf_agents.trajectories.TimeStep(step_type=StepType.FIRST, observation=self._state, reward=reward, discount=discount)
        
        elif len(selected_cards) == 2:
            
            for s in available_sets:
                if selected_cards[0] in s and selected_cards[1] in s:
                    reward = 0
        
            return ts.transition( observation=self._state, reward=reward, discount=0.3)
            return tf_agents.trajectories.TimeStep(step_type=StepType.MID, observation=self._state, reward=reward, discount=discount)
        
        else:
            
            if set.check_set(selected_cards[0],selected_cards[1],selected_cards[2]):
                print("found set")
                reward = 1000
            
            else:
                reward = 0
            
        self._update_pygame()
        clear_selection(self.model.game.in_play_cards)
        self.model.update()
        reward = np.array(reward, dtype=np.float32)
        self._update_state(self.model.game.in_play_cards)
        #return tf_agents.trajectories.TimeStep(step_type=StepType.LAST, observation=self._state, reward=reward, discount=discount)
        
        return ts.transition( observation=self._state, reward=reward, discount=0.1)

    
def random_action(cards):
    selected_cards = random.choices(cards, k=3)
    
    for card in selected_cards:
        card.been_clicked = True

def clear_selection(cards):
    for card in cards:
        card.been_clicked = False
   
def find_sets(cards):
    available_sets = []
    for i in range(len(cards)):
        for j in range(i+1,len(cards)):
            for k in range(j+1,len(cards)):
                if set.check_set(cards[i],cards[j],cards[k]):
                    available_sets.append([cards[i],cards[j],cards[k]])
                    #return [cards[i],cards[j],cards[k]]
    return available_sets   

def best_action(cards):
    card_set = find_sets(cards)[0]
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
    steps_to_win = 0
    view = set.View(model, screen)
    
    while not model.game.sets_found >= 2:
        
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                raise SystemExit
        random_action(model.game.in_play_cards)
        steps_to_win += 1
        screen.process(events)
        model.update()
        clear_selection(model.game.in_play_cards)
        
   
        screen.update()
        screen.render()

        view.draw()
        pygame.display.flip()
        time.sleep(0.001)
    print(steps_to_win)
    pygame.quit()
    
    
def get_rb_observer(replay_buffer_capacity):
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
    return rb_observer, replay_buffer


def collect_episode(environment, policy, num_episodes, rb_observer):

  driver = py_driver.PyDriver(
    environment,
    py_tf_eager_policy.PyTFEagerPolicy(
      policy, use_tf_function=True),
    [rb_observer],
    max_episodes=num_episodes)
  initial_time_step = environment.reset()
  driver.run(initial_time_step)

def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]


# THE MAIN LOOP
if __name__ == "__main__":
   # run_game()
    env = SetEnv(render=True)
#    utils.validate_py_environment(env, episodes=5)

    fc_layer_params = (100,)
    learning_rate = 0.01
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
    
    rb_observer, replay_buffer = get_rb_observer(replay_buffer_capacity = 20000)
    
   # collect_episode(env, tf_agent.collect_policy, 2, rb_observer)
    
    
    num_eval_episodes= 10
    collect_episodes_per_iteration = 1
    num_iterations = 10000
    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    tf_agent.train = common.function(tf_agent.train)

    # Reset the train step
#    tf_agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
  #  avg_return = compute_avg_return(env, tf_agent.policy, num_eval_episodes)
  #  returns = [avg_return]

    for _ in range(num_iterations):
        env.reset()
    # Collect a few episodes using collect_policy and save to the replay buffer.
        collect_episode(env, tf_agent.collect_policy, collect_episodes_per_iteration, rb_observer)
        print(steps_to_win)
    # Use data from the buffer and update the agent's network.
        iterator = iter(replay_buffer.as_dataset(sample_batch_size=1))
        trajectories, _ = next(iterator)
        train_loss = tf_agent.train(experience=trajectories)

        replay_buffer.clear()

        #step = tf_agent.train_step_counter.numpy()

        #if step % 1 == 0:
        #    print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        #if step % 1 == 0:
          #  avg_return = compute_avg_return(env, tf_agent.policy, num_eval_episodes)
           # print('step = {0}: Average Return = {1}'.format(step, avg_return))
           # returns.append(avg_return)

    
 
  