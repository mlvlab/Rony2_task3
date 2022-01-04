# Rony2

### How to run task3
    python main.py --set_num set_01 --run_single_task 3

### Setup manual
1. create python conda environment

    - `conda env create -f challenge.yaml`

2. install torch reid

    - `cd task3/lib/reid/deep-person-reid/`

    - `python setup.py develop`

3. install alphapose 

    - `conda install -c anaconda gxx_linux-64`

    - `ln -s /home/ubuntu/anaconda3/envs/challenge/bin/x86_64-conda_cos6-linux-gnu-gcc /home/ubuntu/anaconda3/envs/challenge/bin/gcc `

    - `ln -s /home/ubuntu/anaconda3/envs/challenge/bin/x86_64-conda_cos6-linux-gnu-g++ /home/ubuntu/anaconda3/envs/challenge/bin/g++`

    - `cd task3/lib/alphapose/ `

    - `python setup.py build develop`


4. overwrite weights 
weights (11/6) : `ubuntu@14.49.44.85:/home/ubuntu/ai_grand_challenge_weights/Rony2_weights_1106.zip`
