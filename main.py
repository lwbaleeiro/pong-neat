import pygame
import os
import neat
import pickle

from pong import Game

class PongGame:
    def __init__(self,window, width, height):
        self.game = Game(window, width, height)
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle
        self.ball = self.game.ball

    def test_ai(self, genome, config):

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
                self.game.move_paddle(left=True, up=True)
            if keys[pygame.K_s]:
                self.game.move_paddle(left=True, up=False)

            output = net.activate((self.right_paddle.y, self.ball.y, abs(self.right_paddle.x - self.ball.x)))
            decision = output.index(max(output))

            if decision == 0:
                pass
            elif decision == 1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)

            self.game.loop()
            self.game.draw()
            pygame.display.update()

        pygame.quit()

    def train_ai(self, genome_p1, genome_p2, config):
        net_p1 = neat.nn.FeedForwardNetwork.create(genome_p1, config)
        net_p2 = neat.nn.FeedForwardNetwork.create(genome_p2, config)

        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()

            output_p1 = net_p1.activate((self.left_paddle.y, self.ball.y, abs(self.left_paddle.x - self.ball.x)))
            decision_p1 = output_p1.index(max(output_p1))

            if decision_p1 == 0:
                pass
            elif decision_p1 == 1:
                self.game.move_paddle(left=True, up=True)
            else:
                self.game.move_paddle(left=True, up=False)

            output_p2 = net_p2.activate((self.right_paddle.y, self.ball.y, abs(self.right_paddle.x - self.ball.x)))
            decision_p2 = output_p2.index(max(output_p2))

            if decision_p2 == 0:
                pass
            elif decision_p2 == 1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)

            game_info = self.game.loop()
            self.game.draw()
            pygame.display.update()

            # Caso um dos dois erre a bola encerra o treinamento atual e passa para o proximo.
            if game_info.left_score >= 1 or game_info.right_score >= 1 or game_info.left_hits > 50:
                self.calculate_fitness(genome_p1, genome_p2, game_info)
                break
    
    def calculate_fitness(self, genome_p1, genome_p2, game_info):
        genome_p1.fitness += game_info.left_hits
        genome_p2.fitness += game_info.right_hits

def evaluation_genomes(genomes, config):
    width, height = 700, 500
    window = pygame.display.set_mode((width,height))

    for i, (genome_id_p1, genome_p1) in enumerate(genomes):
        if i == len(genomes) - 1:
            break

        genome_p1.fitness = 0
        for genome_id_p2, genome_p2 in genomes[i + 1:]:
            genome_p2.fitness = 0 if genome_p2.fitness == None else genome_p2.fitness
            game = PongGame(window, width, height)
            game.train_ai(genome_p1, genome_p2, config)

def run_neat(config):
    # Para carregar a partir de um check point descomente a linha a baixo e comente: population = neat.Population(config)
    population = neat.Checkpointer.restore_checkpoint('neat-checkpoint-40')
    #population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    statistics_reporter = neat.StatisticsReporter()
    population.add_reporter(statistics_reporter)
    # Salva o estado e permite reiniciar o algoritimo em um determinado ponto, em numero de gerações. 
    population.add_reporter(neat.Checkpointer(1))

    winner = population.run(evaluation_genomes, 1)
    with open("best.pickle", "wb") as file:
        pickle.dump(winner, file)

def load_best_winner(config):
    width, height = 700, 500
    window = pygame.display.set_mode((width,height))

    with open("best.pickle", "rb") as file:
        winner = pickle.load(file)

    game = PongGame(window, width, height)
    game.test_ai(winner, config) 

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.ini")
    
    config = neat.Config(neat.DefaultGenome, 
                         neat.DefaultReproduction, 
                         neat.DefaultSpeciesSet, 
                         neat.DefaultStagnation, 
                         config_path)

    #run_neat(config)
    load_best_winner(config)