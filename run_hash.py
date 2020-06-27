import sys
import os
from multiprocessing import Process
import ipdb

'''
run a batch of the worker to speed up
python run_hash.py ${gpu_ids} ${worker_num}
e.g. python run_hash.py 0,1,2,3 12; which will run 3(12/4) worker on each GPU
'''

def obtain_parameters():
    cuda = sys.argv[1]
    worker = int(sys.argv[2])
    gpus = cuda.split(',')
    group_num = int(worker / len(gpus))
    gpu4worker = gpus * group_num
    return gpu4worker, worker

def run_cmd(cmd):
    os.system(cmd)

if __name__ == "__main__":
    gpu4worker, worker = obtain_parameters()
    for gpu, wid in zip(gpu4worker, range(worker)):
        cmd = f'CUDA_VISIBLE_DEVICES={gpu} python -m utils.hash_positive_generate --gpu_id {gpu} --dataset zh50w --model bertretrieval --output data/zh50w/hash --worker {worker} --current_worker {wid}'
        print(f'[!] running the following command:\n{cmd}\n')
        p = Process(target=run_cmd, args=(cmd,))
        p.start()
