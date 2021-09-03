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
        ```
* **Source Code**

    - If you want to train the whole network from begining using source code on the web fine-grained dataset, please follow subsequent steps
    
      - Choose a dataset, create soft link to dataset by
       ```
       ln -s web-bird bird
       ```

      - Modify `CUDA_VISIBLE_DEVICES` to proper cuda device id and `data_base` to proper dataset in  ``` run_train_resnet.sh ```
      
      - Activate virtual environment(e.g. conda) and then run the script ```bash run_train_resnet.sh``` to train a resnet50 model.
      
* **Attentions**     
    - The threshold _thr_ should be set according to the histogram of angles. 60 or 61 is a good choice for CUB200, yet may not be the best.
    - Each step has effects on the final performance, especially step 2. If some erorr occurs in one step, the final performance will drop.
    ![table](performance.png)
