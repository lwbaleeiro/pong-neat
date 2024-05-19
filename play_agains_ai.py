import pygame
import os
import neat
import pickle

from pong import Game

def test_ai(game, genome, config):

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    run = True
    clock = pygame.time.Clock()
    while run:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            game.move_paddle(left=True, up=True)
        if keys[pygame.K_s]:
            game.move_paddle(left=True, up=False)

        output = net.activate((game.right_paddle.y, game.ball.y, abs(game.right_paddle.x - game.ball.x)))
        decision = output.index(max(output))

        if decision == 0:
            pass
        elif decision == 1:
            game.move_paddle(left=False, up=True)
        else:
            game.move_paddle(left=False, up=False)

        game.loop()
        game.draw()
        pygame.display.update()

    pygame.quit()

def load_best_winner(config):
    width, height = 700, 500
    window = pygame.display.set_mode((width,height))

    with open("best.pickle", "rb") as file:
        winner = pickle.load(file)

    game = Game(window, width, height)
    test_ai(game, winner, config) 

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.ini")
    
    config = neat.Config(neat.DefaultGenome, 
                         neat.DefaultReproduction, 
                         neat.DefaultSpeciesSet, 
                         neat.DefaultStagnation, 
                         config_path)
    
    load_best_winner(config)