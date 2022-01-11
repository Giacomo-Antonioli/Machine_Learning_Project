# Machine Learning Project Documentation

<p>
The following file contains the documentation regarding the classes created for the ML Project 2021/2022.
</p>

<p>
<h2>
Classes documentation list:
</h2>
<h3>


1. [Main.py](./docs/mainDoc.md) 
2. [Model.py](./docs/ModelDoc.md)
3. [Model_selection.py](./docs/model_selectionDoc.md)
4. [Layer.py](./docs/layerDoc.md)
5. [Optimizer.py](./docs/OptimizersDoc.md)
6. [Metrics.py](./docs/metricsDoc.md)
7. [LoadCVSData.py](./docs/loadCSVDataDoc.md)
8. [Activations.py](./docs/activations.md)
9. [Loss.py](./docs/loss.md)
10. [Monk.py](./docs/monk.md)
11. [Reguralizers.py](./docs/reguralizers.md)
12. [Weight_Initializer.py](./docs/weightInizializer.md)

</h3>

</p>

<p>
<h2>
 Quick start: 
</h2>

The following section of the document is a guide to install and clone this project.
For a fully functional project, before cloning and starting it some python libraries must be installed.

<h3><strong>numpy</strong></h3>  

 To install numpy use the following commands in a terminal.

PIP

    If you use pip, you can install NumPy with:

    - pip install numpy



CONDA

    If you use conda, you can install NumPy from the defaults or conda-forge channels:

    # Best practice, use an environment rather than install in the base env
    - conda create -n my-env
    - conda activate my-env
    # If you want to install from conda-forge
    - conda config --env --add channels conda-forge
    # The actual install command
    - conda install numpy


 If you have any problems installing numpy, check this [numpy guide to installation](https://numpy.org/install/)  

<h3><strong>pandas</strong></h3>  

To install pandas use the following commands in a terminal.  

On Windows

    - pip install pandas
 
On Ubuntu
    
    - sudo apt-get install python3-pandas

If you have any problems installing pandas, check this [pandas guide to installation](https://pandas.pydata.org/docs/getting_started/install.html)

<h3><strong>scikit_learn</strong></h3>  

To install scikit-learn use the following commands in a terminal.  

PIP

    - pip install -U scikit-learn
 
If you have any problems installing scikit-learn, check this [scikit-learn guide to installation](https://scikit-learn.org/stable/install.html)

<h3><strong>tqdm</strong></h3>  

To install tqdm use the following commands in a terminal.  

PIP

    - pip install tqdm

CONDA
    
    - conda install -c conda-forge tqdm

If you have any problems installing tqdm, check this [tqdm guide to installation](https://pypi.org/project/tqdm/)


<h3><strong>wandb</strong></h3>  

To use wandb you must download and install Docker first: [get docker here](https://www.docker.com/products/docker-desktop)  
You will need a wandb account as well: [signUp to wandb here](https://wandb.ai/site)  

To install wandb use the following commands in a terminal.  
PIP

    - pip install wandb
   
     
To make wandb work on local server write the following in a terminal:

    - wandb local

Behind the scenes the wandb client library is running the wandb/local docker image, forwarding port 8080 to the host, and configuring your machine to send metrics to your local instance instead of our hosted cloud. If you want to run our local container manually, you can run the following docker command:

    - docker run --rm -d -v wandb:/vol -p 8080:8080 --name wandb-local wandb/local

If you have any problems installing wandb, check this [wandb guide to installation](https://docs.wandb.ai/quickstart)
If you have any problems running wandb on local server, check this [wandb locl server guide](https://docs.wandb.ai/guides/self-hosted/local)

</p>


<p>
<h2>
 Software used
</h2>


To organize this project development and to make a better list of features in development, features that will be
developed, changeLogs and other annotations the git hub project board
at: <a href="https://github.com/Giacomo-Antonioli/Machine_Learning_Project/projects/1">ML-CM Project Board</a>


To develop this project the following has been used:

| Software kind     | Software version     |
| ----------------  | -------------------  | 
| Language          | Python 3.9.9         |
| Python Library    | numpy==1.18.5        |
| Python Library    | pandas==1.2.4        |
| Python Library    | scikit_learn==1.0.1  |
| Python Library    | tqdm==4.59.0         |
| Python Library    | wandb==0.12.7        |

</p>