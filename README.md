# Robust Learning from Noisy Web Images via Data Purification for Fine-Grained Recognition

Introduction
------------
This is the source code for our paper **Robust Learning from Noisy Web Images via Data Purification for Fine-Grained Recognition**

Network Architecture
--------------------
The architecture of our proposed model is as follows
![network](network.png)

Installation
------------
After creating a virtual environment of python 3.7, run `pip install -r requirements.txt` to install all dependencies

How to use
------------
The code is currently tested only on GPU
* **Data Preparation**
    - Download data into project root directory and uncompress them using
        ```
        wget https://wsnfg.oss-cn-hongkong.aliyuncs.com/web-bird.tar.gz
        tar -xvf web-bird.tar.gz
      
        # optional
        wget https://wsnfg.oss-cn-hongkong.aliyuncs.com/web-car.tar.gz
        wget https://wsnfg.oss-cn-hongkong.aliyuncs.com/web-aircraft.tar.gz
        tar -xvf web-car.tar.gz
        tar -xvf aircraft-car.tar.gz
        ```
* **Source Code**

    - If you want to train the whole network from begining using source code on the web fine-grained dataset, please follow subsequent steps
    
      - Choose a dataset, create soft link to dataset by
       ```
       ln -s web-bird bird
      
       # optional
       ln -s web-car car
       ln -s web-aircraft aircraft
       ```

      - Modify `CUDA_VISIBLE_DEVICES` to proper cuda device id and `data_base` to proper dataset in  ``` run_train_resnet.sh ```
      
      - Activate virtual environment(e.g. conda) and then run the script ```bash run_train_resnet.sh``` to train a resnet50 model.
      
    ![table](performance.png)
