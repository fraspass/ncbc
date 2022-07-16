import yaml
import os
import pickle
import argparse
from lda_clust import topic_model

if __name__ == "__main__":

    parser = argparse.ArgumentParser("YAML Simulation topic model parser")
    parser.add_argument("--yml-dir", type = str, metavar="S", help = "Root dir for the configuration yaml file")
    args = parser.parse_args()

    # Load yaml configuration file
    with open(args.yml_dir) as file:
        cfg = yaml.safe_load(file)

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
        model.random_init(K_init=cfg["k_init"], H_init=cfg["h_init"])
    elif cfg["mod_init"] == "gensim":
        model.gensim_init(
            chunksize=cfg["chunk"], passes=cfg["passes"], 
            iterations=cfg["gen_iter"], eval_every=cfg["eval_step"], 
            K_init=cfg["k_init"], H_init=cfg["h_init"])
    elif cfg["mod_init"] == "spectral":
        model.spectral_init(K_init = cfg["k_init"], H_init = cfg["h_init"])

    mcmc = model.MCMC(
        iterations=cfg["iterations"], burnin=cfg["burn"], size = cfg["size"],
        verbose=cfg["verbose"],calculate_ll=cfg["calc_ll"],random_allocation=cfg["rand_alloc"],
        jupy_out=cfg["jupy_out"],return_s=cfg["return_s"], return_t=cfg["return_t"],return_z=cfg["return_z"],
        thinning=cfg["thinning"]
        )

    if not os.path.exists("./results/"):
        os.mkdir("./results/")

    with open(cfg["save_dir"],"wb") as savefile:
        pickle.dump(mcmc,savefile,protocol=pickle.HIGHEST_PROTOCOL)
    print(mcmc)


