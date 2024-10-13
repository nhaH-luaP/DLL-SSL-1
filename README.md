# DLL-SSL-1

## Requirements
Installing all dependencies for this repository is a bit complicated. However, you can try setting up our used environment in this specific order.

1. Create a conda env with python 3.10
```bash
conda create -n dll-ssl python=3.10
conda activate dll-ssl
```

2. Install BirdSet
```bash
pip install -e git+https://github.com/DBD-research-group/BirdSet.git#egg=birdset
```

3. Install fairseq
```bash
cd path/to/fairseq/repo
pip install -e .
```
   
4. (Eventually upgrade hydra-core and omegaconf if necessary)
5. Clone EAT repo into the fairseq Repo and install its dependencies
```bash
git clone https://github.com/cwx-worst-one/EAT
cd EAT
pip install -r requirements.txt
```

6. Eventually downgrade numpy
```bash
pip install numpy==1.26.4
```

## Fairseq and Hydra Problem
There is a workaround in this directory to work around problems arising from using a fairseq model and hydra configuration. Therefore, we do not really use hydra, but a config-yaml file that is manually loaded into the main during the run. Therefore, you cannot really work with command-line args but have to adjust configurations manually in the config.yaml file.

<!---
## Open problems that i currently see
- What exactly is mixup doing and does it help? (Implementation from EAT repo, not sure if this is what we are looking for)
- How can we include training on a subset of vocal data here using the birdset
- Is Accuracy correctly calculated? Seems to be on one single value throughout the whole training (or maybe our model is always guessing the same class, which would lead to the same results)
- More thoughts on the accuracy problem. In a multi label setting with few true labels per instance, guessing 0 all the way is a strategy to reach like 95% accuracy but still only
guess randomly. Therefore, accuracy is a bad measure here, maybe try to implement other.
- What are good training parameters to finetune the last layer?
- Slurm Script has to be done but be aware of the Fairseq and Hydra Problem described above!
- Do a comparison of a pretrained backbone versus none.
- If not already clear from the top, I am not at all confident, that the saved requirements.txt will work on the server, so this may be problematic. However, it should be wokring with the envs the devs have prepared themselves.
-->
