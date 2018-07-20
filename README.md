# Playing SNES games with Deep Reinforcement Learning

Some tests for algorithms that try to learn how to play some SNES games using Deep Reinforcement Learning methods.

## Installation

First create a Conda environment using the provided environment file

    conda env create -f environment.yml

After that, enter the environment

    source activate gym-retro

And then you should update your environment to install the latest version of [gym-retro at master](https://github.com/openai/retro/tree/develop#install-from-binary).

Finally you should [install the ROMs](https://github.com/openai/retro/tree/develop#add-new-roms) for the games you want to try.


## Examples

Gradius III

    python policygradientpytorch.py GradiusIII-Snes Level1.Mode1.Shield gradius3.pt

Donkey Kong Country, first level

    python policygradientpytorch.py DonkeyKongCountry-Snes 1Player.CongoJungle.JungleHijinks.Level1 donkeykong.pt

Super Mario World

    python policygradientpytorch.py SuperMarioWorld-Snes DonutPlains1 supermario.pt

### Other platforms games

Sonic the Hedgehog (Genesis)

    python policygradientpytorch.py SonicTheHedgehog-Genesis GreenHillZone.Act1 sonic.pt

Comix Zone (Genesis)

    python policygradientpytorch.py ComixZone-Genesis Episode1.Page1 comixzone.pt
