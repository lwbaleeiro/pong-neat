# NEAT - NeuroEvolution of Augmenting Topologies

You can find more about NEAT in this paper [here](https://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf).

## How to Execute

- First, you need to install the requirements found in the `requirements.txt` file:

  ```bash
  pip install -r requirements.txt

- **If you wants to play** agains the traning ai, simply execute the `play_against_ai.py` file in the project folder, make sure that the project have the `best.pickle` file:

  ```bash
    cd your/folder/path/pong-neat
    python play_against_ai.py

- **If you need to train the ai**, you need to execute the `train_neat.py` once you have the best player (`best.pickle` file) you are done with training.

  ```bash
    cd your/folder/path/pong-neat
    python train_neat.py