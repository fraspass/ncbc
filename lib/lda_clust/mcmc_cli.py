import argparse
import pickle
import os 
from lda_clust.topic_model import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Topic model parser")


    # Topic model constructor parameters
    parser.add_argument("--dir", type = str, metavar="S", help = "Root dir for the data file (pickle)")
    parser.add_argument("--K", type = int, metavar="N", help = "Max Number of session topics")
    parser.add_argument("--fx-k", default=False, action='store_true', help = "Dirichlet distribution prior flag (sessions)")
    parser.add_argument("--H", type = int, default = 0, metavar="N", help = "Max Number of command topics")
    parser.add_argument("--fx-h", default = False, action='store_true', help = "Dirichlet distribution prior flag (commands)")
    parser.add_argument("--V", type = int, default = 0, metavar="N", help = "Number of words in the vocabulary")
    parser.add_argument("--fx-v", default = False, action='store_true', help = "Dirichlet distribution prior flag (words)")
    parser.add_argument("--sec-top", default = False, action='store_true', help = "Secondary topic flag")
    parser.add_argument("--cmd-top", default = False, action='store_true', help = "Command level topic flag")
    parser.add_argument("--gamma", type = float, default = 1.0, metavar="N", help = "Gamma parameter")
    parser.add_argument("--eta", type = float, default = 1.0, metavar="N", help = "Eta")
    parser.add_argument("--alpha", type = float, default = 1.0, metavar="N", help = "Alpha")
    parser.add_argument("--alpha-zero", type = float, default = 1.0, metavar="N", help = "Alpha zero")
    parser.add_argument("--tau", type = float, default = 1.0, metavar="N", help = "Tau")


    #MCMC function parameters
    parser.add_argument("--iter", type = int, metavar="N", help = "Number of iterations")
    parser.add_argument("--burn", type = int, metavar="N", default = 0, help = "Number of burning steps")
    parser.add_argument("--size", type = int, metavar="N", default=1, help = "resample size")
    parser.add_argument("--verbose", default=False, action='store_true', help = "Verbosity")
    parser.add_argument("--calc-ll", default=False, action='store_true', help = "Calculate marginal likelihood")
    parser.add_argument("--rand-alloc", default=False, action='store_true', help = "Random allocation of docs/commands in topics")
    parser.add_argument("--jupy-out", default=False, action='store_true', help = "Jupy out")
    parser.add_argument("--return-t", default=False, action='store_true', help = "Save posterior draws for t")
    parser.add_argument("--return-s", default=False, action='store_true', help = "Save posterior draws for s")
    parser.add_argument("--return-z", default=False, action='store_true', help = "Save posterior draws for z")
    parser.add_argument("--thinning", type = int, default=1, metavar="N", help = "Thinning")


    #Initialization functions
    parser.add_argument(
        "--mod-init", type = str, metavar="S", default = "custom",
        help = 'Initialize models paramters (values: "custom", "random", "gensim", "spectral"})')
    
    # Only for gensim
    parser.add_argument("--chunk", type=int, metavar='N', default=2000, help="Chunksize")
    parser.add_argument("--passes", type=int, metavar='N', default=100, help="Passes")
    parser.add_argument("--gen-iter", type=int, metavar='N', default=100, help="Gensim iterations")
    parser.add_argument("--eval-step", type=int, default=None, help = "Evaluation steps for Gensim")


    # Initial K and H (common for sptectral_init, gensim_init and random_init)
    parser.add_argument("--K-init", type=int, default=None, help = "Number of session topics for initalisation")
    parser.add_argument("--H-init", type=int, default=None, help = "Number of command topics for initalisation")    

    # Results root dir
    parser.add_argument("--save-dir", type = str, metavar="S", help = "Root dir to store the MCMC results")
    
    args = parser.parse_args()
    
    if args.mod_init not in {"custom", "random", "gensim", "spectral"}:
        raise ValueError("Model initialization method is invalid.")

    with open(args.dir, "rb") as file:
        data = pickle.load(file)

    model = topic_model(
        W = data["w"], K = args.K, fixed_K=args.fx_k, 
        H=args.H, fixed_H=args.fx_h, V=args.V, fixed_V=args.fx_v,
        secondary_topic=args.sec_top, command_level_topics=args.cmd_top,
        gamma=args.gamma, eta=args.eta, alpha=args.alpha, alpha0=args.alpha_zero, tau=args.tau)
    
    if not args.sec_top:
        data.update({"z": None})
    if not args.cmd_top:
        data.update({"s": None})
    
    if args.mod_init == "custom":
        model.custom_init(t = data["t"], s = data["s"], z = data["z"])
    elif args.mod_init == "random":
        model.random_init(K_init=args.K_init, H_init=args.H_init)
    elif args.mod_init == "gensim":
        model.gensim_init(
            chunksize=args.chunk, passes=args.passes, 
            iterations=args.gen_iter, eval_every=args.eval_step, 
            K_init=args.K_init, H_init=args.H_init)
    elif args.mod_init == "spectral":
        model.spectral_init(K_init = args.K_init, H_init = args.H_init)

    mcmc = model.MCMC(
        iterations=args.iter, burnin=args.burn, size = args.size,
        verbose=args.verbose,calculate_ll=args.calc_ll,random_allocation=args.rand_alloc,
        jupy_out=args.jupy_out,return_s=args.return_s, return_t=args.return_t,return_z=args.return_z,
        thinning=args.thinning
        )

    #if not os.path.exists("./results/"):
    #    os.mkdir("./results/")

    #with open(args.save_dir,"wb") as savefile:
    #    pickle.dump(mcmc,savefile,protocol=pickle.HIGHEST_PROTOCOL)
    print(mcmc)
    