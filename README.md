# Setup

## VM Setup:
We've run all the code on a GCP Ubuntu 20.04 VM with 8 V100 GPU, Cuda 11.7, CuDNN 8.5
setup script can be seen here : 
https://gist.githubusercontent.com/sourabh2k15/9dbadd0f5ca35568ca210ee4cb3b19c1/raw/be988210438f91eda85fc608a9c71744a4c8af89/vm_setup.sh

we simply wget above script and run to install required elements. 

## Code Setup 
git clone https://github.com/sourabh2k15/deepspeech_timing_diffs.git

cd deepspeech_timing_diffs

## Data Setup 
We use git lfs to store a real 256 size librispeech batch with input and target paddings. 
In case it's not present when you git clone, please manually download using :
```
wget https://transfer.sh/oanN8q/sharded_padded_batch.npz
```

# Reproduction

## JAX
Command to run the jax example : 
```
python3 jax/jax_e2e.py
```

## PyTorch
Command to run the torch example : 
```
torchrun --standalone --nnodes 1 --nproc_per_node 8 pytorch/torch_e2e.py
```

# Results: 


1) on 8 V100 gcp VM torch script gives ~80 seconds for 100 steps which translates to 1.25 steps / second
this is in line with the e2e run here : https://tensorboard.dev/experiment/aXqps4I4QlG90gfMRoAaJg/#scalars&_smoothingWeight=0


2) JAX script gives 190 seconds for 10 steps which is much much slower than the full e2e run here which shows a 0.74 steps / second : https://tensorboard.dev/experiment/BOSgqsQGTGugYJ1pDnPHbw/#scalars&_smoothingWeight=0


3) above jax e2e script is very similar to e2e run so slowdown is unexpected here.  
