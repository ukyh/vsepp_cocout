# Improving Visual-Semantic Embeddings with Hard Negatives

The code for `R@K` evaluation of our paper, **[Switching to Discriminative Image Captioning by Relieving a Bottleneck of Reinforcement Learning](https://github.com/ukyh/switch_disc_caption.git)** (WACV 2023).

### Acknowledgment
The code is based on [vsepp](https://github.com/fartashf/vsepp).
We thank the authors of the repository.


## Setup

```bash
git clone https://github.com/ukyh/vsepp_cocout.git
cd vsepp_cocout

conda create --name vsepp3 python=3.6
conda activate vsepp3

pip install torch==1.4.0 torchvision==0.5.0
pip install tensorboard
conda install -c conda-forge pycocotools
pip install nltk
python -c "import nltk; nltk.download('punkt')"
```


## Downloads

```bash
wget http://www.cs.toronto.edu/~faghri/vsepp/vocab.tar
tar -xvf vocab.tar 
wget http://www.cs.toronto.edu/~faghri/vsepp/data.tar
tar -xvf data.tar
wget http://www.cs.toronto.edu/~faghri/vsepp/runs.tar
tar -xvf runs.tar
rm *.tar

cp vocab/* data

cd data/coco
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
rm images
mkdir images 
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip -d images
rm *.zip
```


## Run

Copy the output files to evaluate from [switch_disc_caption](https://github.com/ukyh/switch_disc_caption) (the files under `eval_results`).  
Then, run the following commands.
The output is formatted as:  
`Text to image`: `R@1` `R@5` `R@10` `...`

```bash
cd vsepp_cocout
conda activate vsepp3

DATA_PATH=data
RUN_PATH=runs
# val or test
SPLIT=test
GEN_PATH=eval_results/sample_test.json

python -c "\
from vocab import Vocabulary
import evaluation_gen as evaluation
evaluation.evalrank('$RUN_PATH/coco_vse++_resnet_restval_finetune/model_best.pth.tar', data_path='$DATA_PATH', gen_path='$GEN_PATH', split='${SPLIT}')"
```

<!-- Code for the image-caption retrieval methods from
**[VSE++: Improving Visual-Semantic Embeddings with Hard Negatives](https://arxiv.org/abs/1707.05612)**
*, F. Faghri, D. J. Fleet, J. R. Kiros, S. Fidler, Proceedings of the British Machine Vision Conference (BMVC),  2018. (BMVC Spotlight)*

## Dependencies
We recommended to use Anaconda for the following packages.

* Python 3.5
* [PyTorch](http://pytorch.org/) (>0.2)
* [NumPy](http://www.numpy.org/) (>1.12.1)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)
* [pycocotools](https://github.com/cocodataset/cocoapi)
* [torchvision]()
* [matplotlib]()


* Punkt Sentence Tokenizer:
```python
import nltk
nltk.download()
> d punkt
```

## Download data

Download the dataset files and pre-trained models. We use splits produced by [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/deepimagesent/). The precomputed image features are from [here](https://github.com/ryankiros/visual-semantic-embedding/) and [here](https://github.com/ivendrov/order-embedding). To use full image encoders, download the images from their original sources [here](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html), [here](http://shannon.cs.illinois.edu/DenotationGraph/) and [here](http://mscoco.org/).

```bash
wget http://www.cs.toronto.edu/~faghri/vsepp/vocab.tar
wget http://www.cs.toronto.edu/~faghri/vsepp/data.tar
wget http://www.cs.toronto.edu/~faghri/vsepp/runs.tar
```

We refer to the path of extracted files for `data.tar` as `$DATA_PATH` and 
files for `models.tar` as `$RUN_PATH`. Extract `vocab.tar` to `./vocab` 
directory.

*Update: The vocabulary was originally built using all sets (including test set 
captions). Please see issue #29 for details. Please consider not using test set 
captions if building up on this project.*

## Evaluate pre-trained models

```python
DATA_PATH=data
RUN_PATH=runs
python -c "\
from vocab import Vocabulary
import evaluation
evaluation.evalrank('$RUN_PATH/coco_vse++/model_best.pth.tar', data_path='$DATA_PATH', split='test')"
``` -->

<!-- ## Training new models
Run `train.py`:

```bash
python train.py --data_path "$DATA_PATH" --data_name coco_precomp --logger_name 
runs/coco_vse++ --max_violation
```

Arguments used to train pre-trained models:

| Method    | Arguments |
| :-------: | :-------: |
| VSE0      | `--no_imgnorm` |
| VSE++     | `--max_violation` |
| Order0    | `--measure order --use_abs --margin .05 --learning_rate .001` |
| Order++   | `--measure order --max_violation` | -->


## Reference

If you found this code useful, please cite the following papers:

    @inproceedings{honda2023switch,
      title={Switching to Discriminative Image Captioning by Relieving a Bottleneck of Reinforcement Learning},
      author={Honda, Ukyo and Taro, Watanabe and Yuji, Matsumoto},
      booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
      year={2023}
    }

    @article{faghri2018vse++,
      title={VSE++: Improving Visual-Semantic Embeddings with Hard Negatives},
      author={Faghri, Fartash and Fleet, David J and Kiros, Jamie Ryan and Fidler, Sanja},
      booktitle = {Proceedings of the British Machine Vision Conference ({BMVC})},
      url = {https://github.com/fartashf/vsepp},
      year={2018}
    }

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
