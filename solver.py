import set
import pygame
import planes
import planes.gui
import time
import random
import numpy as np
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700

BLACK = (0, 0, 0)
class QAgent():
    def __init__(self):
        
        self.n_states = (1,3**4,2)  # 4 options + already selected for 12 cards
        self.n_actions = (12,) # TODO: Wif nothing works, problem!

        self.qtable = np.zeros(self.n_states + self.n_actions)
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
    def get_state(self,cards):
        state = []
        for i,card in enumerate(cards):
            card_state = 0
            card_state += set.colors.index(card.color)
            card_state += set.shapes.index(card.shape) * 3
            card_state += set.numbers.index(card.number) * 9
            card_state += set.shades.index(card.shade) * 27
            clicked =  1 if card.been_clicked else 0
            state.append([i,card_state,clicked])
        return np.array(state)
    
    
    
    def get_action(self,cards):
        state = self.get_state(cards)
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.n_actions[0])
        else:
            action = np.argmax(self.qtable[state])
        cards[action].been_clicked = True
        return action
     
    def perceive_reward(self, reward, action, cards):
        state = self.get_state(cards)
        self.qtable[state + (action,)] = self.qtable[state + (action,)] + self.alpha * (reward + self.gamma * np.max(self.qtable[state]) - self.qtable[state + (action,)])

def random_action(cards):
    selected_cards = random.choices(cards, k=3)
    
    for card in selected_cards:
        card.been_clicked = True

def clear_selection(cards):
    for card in cards:
        card.been_clicked = False
        
def evualate_selection(cards):
    selected = []
    for card in cards:
        if card.been_clicked:
            selected.append(card)
    if len(selected) != 3:
        return -100
    if set.check_set(selected[0], selected[1], selected[2]):
        return 100
    return -20
def run_game():
    pygame.init()
    size = (WINDOW_WIDTH, WINDOW_HEIGHT)
    screen = planes.Display(size)
    screen.grab = False
    agent = QAgent()

    screen.image.fill(BLACK)
    model = set.Model()
    model.mode = 1
    model.game = set.Game(0, model)
    
    view = set.View(model, screen)
    running = True
    while running:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                raise SystemExit
        #sets_found = model.game.sets_found
        #random_action(model.game.in_play_cards)
        prev_action = None
        for i in range(3):
            action = agent.get_action(model.game.in_play_cards)
            if prev_action == action:
                agent.perceive_reward(-100, action, model.game.in_play_cards)
        
        reward = evualate_selection(model.game.in_play_cards)
        print(reward)
        agent.perceive_reward(reward,action,model.game.in_play_cards)
        screen.process(events)
        model.update()
        #if model.game.sets_found > sets_found:
        #    print("found set")
            #print("hi")
        clear_selection(model.game.in_play_cards)
        
   
        screen.update()
        screen.render()

        view.draw()
        pygame.display.flip()
        time.sleep(0.001)

    pygame.quit()

# THE MAIN LOOP
if __name__ == "__main__":
    run_game()