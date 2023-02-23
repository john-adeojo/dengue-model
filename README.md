# dengue-model

## Jupyter Notebook
There is a kedro version of the project and a Jupyter notebook. 
[Jupyter notebook link]( https://mybinder.org/v2/gh/john-adeojo/dengue-model/65cfc857860866f999163c07b172bc35cea6e0f3?urlpath=lab%2Ftree%2Fnotebooks%2FDengue%20Modelling%20Notebook.ipynb)

## Running Kedro Project
Follow these steps to run the project in Kedro. Kedro gives the ability to view the model in modular form, interact and visualise the model pipelines. 

### pre-requisites 
you will need to install [Anaconda](https://www.anaconda.com/) if you don't already have it on your machine

### Step 1:  Open anaconda powershell and run the following commands sequentially:
```
conda create -n dengue-model-env python=3.8
```
```
conda activate dengue-model-env
```
```
cd dengue-model-env
```
### Step 2: clone this github repo by running these commands in the powershell:
```
git clone https://github.com/john-adeojo/dengue-model.git
```
### Step 3: install the required packages by running the following commands sequentially in the powershell.
```
cd dengue-model
```
```
pip install -r src/requirements.txt
```
### Step 4: run the following commands in the powershell:
The first command runs the model end to end.
```
Kedro run
```
The second command produces a GUI for the model that you can interact with 
```
Kedro viz
```
## Kedro

For more information on kedro:
Take a look at the [Kedro documentation](https://kedro.readthedocs.io) to get started.



