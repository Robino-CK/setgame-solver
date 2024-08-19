import set
import pygame
import planes
import planes.gui
import time
import random
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700

BLACK = (0, 0, 0)

def random_action(cards):
    selected_cards = random.choices(cards, k=3)
    
    for card in selected_cards:
        card.been_clicked = True

def clear_selection(cards):
    for card in cards:
        card.been_clicked = False
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
    running = True
    while running:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                raise SystemExit
        sets_found = model.game.sets_found
        random_action(model.game.in_play_cards)
        
        screen.process(events)
        model.update()
        if model.game.sets_found > sets_found:
            print("found set")
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