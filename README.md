# Decomposed Fusion with Soft Prompt (DFSP)
DFSP is a model which decomposes the prompt language feature into state feature and object feature, then fuses them with image feature to improve the response for state and object respectively.


## Setup
```
conda create --name clip python=3.7
conda activate clip
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install git+https://github.com/openai/CLIP.git
```
Alternatively, you can use `pip install -r requirements.txt` to install all the dependencies.

## Download Dataset
We experiment with three datasets: MIT-States, UT-Zappos, and C-GQA.
```
sh download_data.sh
```

If you already have setup the datasets, you can use symlink and ensure the following paths exist:
`data/<dataset>` where `<datasets> = {'mit-states', 'ut-zappos', 'cgqa'}`.


## Training
```
python -u train.py --dataset <dataset>
```
## Evaluation
We evaluate our models in two settings: closed-world and open-world.

### Closed-World Evaluation
```
python -u test.py --dataset <dataset>
```
You can replace `--dataset` with `{mit-states, ut-zappos, cgqa}`.


### Open-World Evaluation
For our open-world evaluation, we compute the feasbility calibration and then evaluate on the dataset.

### Feasibility Calibration
We use GloVe embeddings to compute the similarities between objects and attributes.
Download the GloVe embeddings in the `data` directory:

```
cd data
wget https://nlp.stanford.edu/data/glove.6B.zip
```
Move `glove.6B.300d.txt` into `data/glove.6B.300d.txt`.

To compute feasibility calibration for each dataset, run the following command:
```
python -u feasibility.py --dataset mit-states
```
The feasibility similarities are saved at `data/feasibility_<dataset>.pt`.

To run, just edit the open-world parameter in config/<dataset>.yml



## References
If you use this code, please cite
```
@article{lu2022decomposed,
  title={Decomposed Soft Prompt Guided Fusion Enhancing for Compositional Zero-Shot Learning},
  author={Lu, Xiaocheng and Liu, Ziming and Guo, Song and Guo, Jingcai},
  journal={arXiv preprint arXiv:2211.10681},
  year={2022}
}
```