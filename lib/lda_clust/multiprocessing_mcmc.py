from audioop import mul
import multiprocessing
import yaml
import os
import pickle
import argparse
from topic_model import *

from multiprocessing import Queue, Process, current_process, freeze_support




parser = argparse.ArgumentParser("Multiproessing parser for simulations")

# Topic model constructor parameters
parser.add_argument("--yml-dir", type = str, metavar="S", help = "yaml config file directory")
parser.add_argument("--proc", type = int, metavar="N", help = "Number of cores")
parser.add_argument("--jobs", type = int, metavar="N", help = "Number of jobs")

args = parser.parse_args()

# Yaml directory    
with open(args.yml_dir) as file:
    cfg = yaml.safe_load(file)

def init_mcmc(cfg):
    
    with open(cfg["data_dir"], "rb") as file:
        data = pickle.load(file)
    
    model = topic_model(
        W = data["w"], K = cfg["K"], fixed_K=cfg["fixed_K"], 
        H=cfg["H"], fixed_H=cfg["fixed_H"], V=cfg["V"], fixed_V=cfg["fixed_V"],
        secondary_topic=cfg["secondary_topic"], command_level_topics=cfg["command_level_topics"],
        gamma=cfg["gamma"], eta=cfg["eta"], alpha=cfg["alpha"], alpha0=cfg["alpha_zero"], tau=cfg["tau"])

    if not cfg["secondary_topic"]:
        data.update({"z": None})
    if not cfg["command_level_topics"]:
        data.update({"s": None})
    
    if cfg["mod_init"] == "custom":
        model.custom_init(t = data["t"], s = data["s"], z = data["z"])
    elif cfg["mod_init"] == "random":
        model.random_init(K_init=cfg["K_init"], H_init=cfg["H_init"])
    elif cfg["mod_init"] == "gensim":
        model.gensim_init(
            chunksize=cfg["chunk"], passes=cfg["passes"], 
            iterations=cfg["gen_iter"], eval_every=cfg["eval_step"], 
            K_init=cfg["K_init"], H_init=cfg["H_init"])
    elif cfg["mod_init"] == "spectral":
        model.spectral_init(K_init = cfg["K_init"], H_init = cfg["H_init"])

    mcmc = model.MCMC(
        iterations=cfg["iterations"], burnin=cfg["burn"], size = cfg["size"],
        verbose=cfg["verbose"],calculate_ll=cfg["calc_ll"],random_allocation=cfg["rand_alloc"],
        jupy_out=cfg["jupy_out"],return_s=cfg["return_s"], return_t=cfg["return_t"],return_z=cfg["return_z"],
        thinning=cfg["thinning"],return_change_s=cfg["return_change_s"], return_change_t=cfg["return_change_t"],
        return_change_z=cfg["return_change_z"]
    )

    return mcmc

def worker(input, output): # !! Meaning of "STOP" : when "STOP" found in input (no more a tuple, it's going to be just a string "STOP" in task_queue)
    for func, args in iter(input.get, "STOP"): # iter makes Queue() (task queue specifically in this case) iterable as otherwise it's not, each item in task queue is tuple
        result = calculate_mcmc(func, args)
        output.put(result)

def calculate_mcmc(fun, args): # executes the function (fun argument which corresponds to first element of tuple in task_queue) with arguments the second element of tuple in task_queue
    result = fun(args)
    return (result, args)

def main(num_proc:int = 2, jobs:int = 2):
    print(cfg)
    NUMBER_PROCESSES = min(num_proc, multiprocessing.cpu_count() - 1) #core number
    CONFIGURATIONS = []
    JOBS = jobs

    task_queue = Queue()
    done_queue = Queue()


    for i in range(JOBS):
        task = {}
        for key,val in cfg.items():
            try:
                task[key] = val[i]
            except:
                task[key] = None
        CONFIGURATIONS.append(task)
    #print(CONFIGURATIONS)

    # Creates the tasks being a tuple consisting of the function to be executed and the corresponding job from config
    TASKS = [(init_mcmc, i) for i in CONFIGURATIONS]

    for task in TASKS: # Create a queue of tasks by puting each job (TASKS) in queue
        task_queue.put(task)

    for i in range(NUMBER_PROCESSES):
        Process(target=worker, args=(task_queue, done_queue)).start() # runs init_mcmc which is in calculate_mcmc which is in worker

    for i in range(JOBS):
        results = done_queue.get()
        file_name = cfg["save_dir"][i]
        with open(file_name, "wb") as file:
            pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Done iter {str(i)}")

    for i in range(JOBS):
        task_queue.put("STOP")



if __name__ == "__main__":
    freeze_support()
    main(num_proc = args.proc, jobs=args.jobs)

    