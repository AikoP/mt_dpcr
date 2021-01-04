# Application of Geometric DeepLearning for Point Cloud Restoration

This repository serves as a codebase for the my master's thesis.

**Note:**: 
The code provided is still in a development stage may be instable.
Requirements include
* CUDA 10.1
* Python 3.6.2
* PyTorch 1.5.1 (!)
* Numpy 1.16.4

## Installation & General

All scripts are supposed to be run in the root folder `/mt_dpcr`.

### a. Cloning

Clone Repository:

```git
git clone https://github.com/AikoP/mt_dpcr
cd dpcr
```

### b. Data Generation

The preferred way is to add paths to the `utils/generator.py` file and then call the generator as follows:

```python
python ./utils/generator.py --dataset=multi_faces --train_size=200 --test_size=20 --val_size=100 --h_min=3 --h_max=16 --n_min=3 --n_max=11
```

This will generate a training/test/validation file in the base directory of the given dataset.

### c. Training

After having generated the train file, training can be started by running

```python
python ./training/train_detector.py --dataset=multi_faces --epochs=10 --batch_size=4 --lr=0.0005 --model=cnet
python ./training/train_corrector.py --dataset=multi_faces --epochs=10 --batch_size=4 --lr=0.0005 --model=cnet
```

This will generate checkpoints after each epoch in `training/checkpoints/../../`.
A run.log is also be generated with full log details (live logging).
The resulting checkpoint contains all information about the training settings, so training can be resumed later if needed.
The checkpoint contains also the model parameters, so you can easily load a model from a checkpoint.

### c. Loading Models

To load a model, simply import the module and use the load function.
This will automatically extract the required model settings.

```python
from models import models
predictor = models.loadModel('../path/to/predictor_checkpoints.t7', device=device)
detector = models.loadModel('../path/to/detector_checkpoints.t7', device=device)
```

### c. Pre-trained Models

Our checkpoints contain model states for each epoch, so unfortunately they are too large upload to GitHub (100MB per-file restriction).
Instead we have extracted the latest model state dict for each of our trained models and made it available under `models/pretrained`.
This way you can manually load the model state dictionary.

## Inference

The models accept (b x d x n) input tensor, with
* b = batchsize
* d = input channels (dimensions)
* n = number of points/vertices

## Generating test data

Note the paths to your input models and import the `utils/generator.py` module.
In the following `val_model_paths` is a list containing the file paths as strings.

```python
from utils import generator
from models import models

print ("\nReading models.. \n")
models, knns = generator.readModels(val_model_paths, device=device, incrementKNNid = False)

print ("\nGenerating samples for %d models.." % (len(models)))
start = time.time()
samples, sample_stats = generator.getSamples(knns, 100, hidden_range = (3,6), iteration_range = (3,6))
print ("   ..done! (%.1fs) ---> #samples: %d" % (time.time() - start, len(samples)))
```

This generates roughly 450 samples for each input model.
See `evaluation/cross_validation.ipynb` for more details.

## Reconstruction

For reconstruction, you need paths to supply a predictor- and detector model.
You also need to import the `utils/reconstructor.py` module and

* A corrector model is optional.
* Make sure to set the `max_iters` paramter.
* `t` is the threshold parameter for the detector model
* if you want logs set `verbose = true`

```python
from utils import reconstructor
output, new_pts_list = reconstructor.reconstruct(sample, predictor, detector, corrector=corrector, corrector_stage=5, max_iters=10, t=0.9, verbose=True)
```

The individual reconstruction steps can be obtained by looking through `new_pts_list`.

############################################################################################################################

I do not own any of the original models. Please see the references to obtain them