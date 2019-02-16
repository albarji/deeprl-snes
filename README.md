# Playing SNES games with Deep Reinforcement Learning

Some tests for algorithms that learn how to play Super Nintendo and other retro games using Deep Reinforcement Learning methods.

## Installation

First create a Conda environment using one of the provided environment file, either for CPU or GPU computation. GPU is highly recommended.

    conda env create -f environment-gpu.yml

After that, activate the environment

    source activate gym-retro-gpu

Finally you should [install the ROMs](https://github.com/openai/retro/tree/develop#add-new-roms) for the games you want to try.

## Training

The ideal hardware to train the player agent is to make use of a machine with a single GPU and a large number of CPUs. For instance, you could use a [g3.4xlarge AWS instance](https://aws.amazon.com/es/blogs/aws/new-next-generation-gpu-powered-ec2-instances-g3/). Be warned that training might take days!

Once you have decided on a game, you should desing a **controller mapping** for such game. This is to easen the learning process: only allow the agent to perform button combinations that make sense in the game. Just take a look at other examples in the [games configuration file](https://github.com/albarji/deeprl-snes/blob/master/games.yaml).

After this is done you can train on the game for a given level by running

    python rllib.py GAME LEVEL --workers CPUS_ON_YOUR_MACHINE

Regular checkpoints will be saved under a folder in your *~/ray-results* directory. You can visualize the training metrics by running

    tensorboard --logdir=LOGDIR

You can then see the agent playing by running

    python rllib.py  GAME LEVEL --test --checkpoint CHECKPOINTFILE
    
Movies of the playthrough can be recorded by adding the `--makemovie` argument.

### Train examples

Gradius III (SNES)

    python rllib.py GradiusIII-Snes Level1.Mode1.Shield

Donkey Kong Country (SNES)

    python rllib.py DonkeyKongCountry-Snes 1Player.CongoJungle.JungleHijinks.Level1

Super Mario World (SNES)

    python rllib.py SuperMarioWorld-Snes DonutPlains1

Sonic the Hedgehog (Genesis)

    python rllib.py SonicTheHedgehog-Genesis GreenHillZone.Act1

Comix Zone (Genesis)

    python rllib.py ComixZone-Genesis Episode1.Page1

### Results

Work in progress! But do check [my Twitter](https://twitter.com/albarjip) for updates.
