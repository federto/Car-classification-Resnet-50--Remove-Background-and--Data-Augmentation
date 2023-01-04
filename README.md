# Sprint Project 05
> Vehicle classification from images

## 1. Install

You can use `Docker` to install all the needed packages and libraries easily. Two Dockerfiles are provided for both CPU and GPU support.

- **CPU:**

```bash
$ docker build -t sp_05 -f docker/Dockerfile .
```

- **GPU:**

```bash
$ docker build -t sp_05 -f docker/Dockerfile_gpu .
```

### Run Docker

- **CPU:**

```bash
docker run --rm --net host -it -v "$(pwd):/home/app/src" --workdir /home/app/src sp_05 bash
```

- **GPU:**

```bash
$ docker run --rm --net host --gpus all -it -v "$(pwd):/home/app/src" --workdir /home/app/src sp_05 bash


```

### Run Unit test


```bash
$ pytest tests/
```
python -m unittest tests/test_api.py

## Working on the project

Below you will the steps and the order in which we think you should solve this project.

### 2. Prepare your data

As a first step, we must extract the images from the file `car_ims.tgz` and put them inside the `data/` folder. Also place the annotations file (`car_dataset_labels.csv`) in the same folder. It should look like this:

```
data/
    ├── car_dataset_labels.csv
    ├── car_ims
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   ├── ...
```

Then, you should be able to run the script `scripts/prepare_train_test_dataset.py`. It will format your data in a way Keras can use for training our CNN model.

You will have to complete the missing code in this script to make it work.

### 3. Train your first CNN (Resnet50)

After we have our images in place, it's time to create our first CNN and train it on our dataset. To do so, we will make use of `scripts/train.py`.

The only input argument it receives is a YAML file with all the experiment settings like dataset, model output folder, epochs,
learning rate, data augmentation, etc.

Each time you are going to train a new model, we recommend you create a new folder inside the `experiments/` folder with the experiment name. Inside this new folder, create a `config.yml` with the experiment settings. We also encourage you to store the model weights and training logs inside the same experiment folder to avoid mixing things between different runs. The folder structure should look like this:

```bash
experiments/
    ├── exp_001
    │   ├── config.yml
    │   ├── logs
    │   ├── model.01-6.1625.h5
    │   ├── model.02-4.0577.h5
    │   ├── model.03-2.2476.h5
    │   ├── model.05-2.1945.h5
    │   └── model.06-2.0449.h5
    ├── exp_002
    │   ├── config.yml
    │   ├── logs
    │   ├── model.01-7.4214.h5
    ...
```

You can check the file `experiments/config_example.yml` to get an idea of all the configurations you can set for an experiment.

The script `scripts/train.py` is already coded but it makes use of external functions from other project modules that you must code to make it work. Mainly, you will have to complete:

- `utils.load_config()`: Takes as input the path to an experiment YAML configuration file, loads it, and returns a dict.
- `resnet50.create_model()`: Returns a CNN ready for training or for evaluation, depending on the input parameters received. Part of coding these functions will require you to create the layers of your first CNN with Keras.
- `data_aug.create_data_aug_layer()`: Used by `resnet50.create_model()`. This function adds data augmentation layers to our model that will be used only while training.

### 4. Evaluate your trained model

After running many experiments and having a potentially good model trained. It's time to check its performance on our test dataset and prepare a nice report with some evaluation metrics.

We will use the notebook `notebooks/Model Evaluation.ipynb` to do it. As you may see, you will have to complete some missing pieces in the notebook to make it run.

Particularly, you will have to complete `utils.predict_from_folder()` function (used in the notebook), which contains the main logic to get predictions from our trained model.

### 5. Improve classification by removing noisy background

As we already saw in the `notebooks/EDA.ipynb` file. Most of the images have a background that may affect our model learning during the training process.

It's a good idea to remove this background. One thing we can do is to use a Vehicle detector to isolate the car from the rest of the content in the picture.

We will use [Detectron2](https://github.com/facebookresearch/detectron2) framework for this. It offers a lot of different models, you can check in its [Model ZOO](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md#faster-r-cnn). We will use for this assignment the model called "R101-FPN".

In particular, we will use a detector model trained on [COCO](https://cocodataset.org) dataset which has a good balance between accuracy and speed. This model can detect up to 80 different types of objects but here we're only interested on getting two out of those 80, those are the classes "car" and "truck".

For this assignment, you will have to complete with the corresponding code the following two files:

- `scripts/remove_background.py`: It will process the initial dataset used for training your model on **item (3)**, removing the background from pictures and storing the resulting images on a new folder.
- `utils/detection.py`: This module loads our detector and implements the logic to get the vehicle coordinate from the image.

Now you have the new dataset in place, it's time to start training a new model and checking the results in the same way as we did for steps items **(3)** and **(4)**.

### 6. Write a report

Finally, we ask you to create and submit with your project a detailed report in Markdown format or Jupyter notebook showing the experiments you did and the results obtained so far.
Efficiently communicating your work is a crucial skill for Machine Learning Engineers in order to correctly show to peers or managers all the work done by you and the improvements obtained.
