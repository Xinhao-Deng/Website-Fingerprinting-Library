# Convert packets to matrix
# python baselines/gen_tam.py -d Undefended
import numpy as np
import os
import argparse
from typing import List
import time
from tqdm import tqdm
from multiprocessing import Process

def gen_matrix(sequences):
    maximum_load_time = 80
    max_matrix_len = 1800
    matrix = []
    for idx in range(sequences.shape[0]):
        cur_X = sequences[idx]
        feature = np.zeros((2, max_matrix_len))
        for pack in cur_X:
            if pack == 0:
                break
            elif pack > 0:
                if pack >= maximum_load_time:
                    feature[0,-1] += 1
                else:
                    idx = int(pack * (max_matrix_len - 1) / maximum_load_time)
                    feature[0,idx] += 1
            else:
                pack = np.abs(pack)
                if pack >= maximum_load_time:
                    feature[1,-1] += 1
                else:
                    idx = int(pack * (max_matrix_len - 1) / maximum_load_time)
                    feature[1,idx] += 1
        matrix.append(feature)

    matrix = np.array(matrix)
    return matrix

def process(file_name):
    data = np.load(os.path.join(inpath, file_name))
    X = data["X"][:,0:args.length]
    print("shape:", X.shape)
    y = data["y"]
    X = gen_matrix(X)
    print(f"{file_name} process done: X = {X.shape}, y = {y.shape}")
    np.savez(os.path.join(inpath, f"rf_tam_{args.length}_{file_name}"), X = X, y = y)

parser = argparse.ArgumentParser(description='Feature extraction')
parser.add_argument("-d", '--dataset', default="Undefended", type=str, help='dataset name')
parser.add_argument("-l", '--length', default=5000, type=int, help='packet length')


seed = 2023
args = parser.parse_args()
inpath = f"datasets/{args.dataset}"

valid_files = []
for file_name in os.listdir(inpath):
    if file_name.startswith("train") or file_name.startswith("test") or file_name.startswith("valid"):
       valid_files.append(file_name)


pool: List[process] = []
for file_name in valid_files:
    p = Process(target=process, args=(file_name,))
    pool.append(p)
    p.start()

while True:
    children_alive = [p.is_alive() for p in pool]
    if not any(children_alive):
        print("all child process done, exit!")
        break

    for i, alive in enumerate(children_alive):
        if not alive:
            print(f"process {i} not alive, args: {valid_files[i]}")

    print("check child process alive done! sleep 10 seconds")
    time.sleep(10)
