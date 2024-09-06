# DLL-SSL-1

## Requirements
Installing all dependencies for this repository is a pain in the butt. You can try to run the following lines of code but im not quite sure this will work...

```
conda create -n dll-ssl
conda activate dll-ssl
pip install -r requirements.txt
```

## Fairseq and Hydra Problem
There is a workaround in this directory to work around problems arising from using a fairseq model and hydra configuration. Therefore, we do not really use hydra, but a config-yaml file that is manually loaded into the main during the run. Therefore, you cannot really work with command-line args but have to adjust configurations manually in the config.yaml file.

## Open problems that i currently see
- What exactly is mixup doing and does it help?
- How can we include training on a subset of vocal data here using the birdset
- Is Accuracy correctly calculated? Seems to be on one single value throughout the whole training (or maybe our model is always guessing the same class, which would lead to the same results)
- More thoughts on the accuracy problem. In a multi label setting with few true labels per instance, guessing 0 all the way is a strategy to reach like 95% accuracy but still only
guess randomly. Therefore, accuracy is a bad measure here, maybe try to implement other.
- What are good training parameters to finetune the last layer?
- Slurm Script has to be done but be aware of the Fairseq and Hydra Problem described above!
- Do a comparison of a pretrained backbone versus none.
- If not already clear from the top, I am not at all confident, that the saved requirements.txt will work on the server, so this may be problematic. However, it should be wokring with the envs the devs have prepared themselves.