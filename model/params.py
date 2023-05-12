#params for this model
import sys

from absl import flags

# ---------- Pre-Processing ----------
flags.DEFINE_integer("shrink_size", 8, "Size for shrinking the origin data.")
flags.DEFINE_string("final_data_dir", "./data/final", "Path to save the pre-process output(as the final data to model).")
flags.DEFINE_list("variables", ["sst","uwind", "vwind", "rain", "vapor", "cloud", ], "The variables for building the model.")

# remote sensing dataset experiment
flags.DEFINE_string("remote_sensing_dataset_dir", "./data/remote_sensing_dataset", "Path to save the origin data.")
flags.DEFINE_string("remote_sensing_npz_dir", "./data/remote_sensing_dataset/final", "Path to save trained models.")
flags.DEFINE_list("remote_sensing_variables", ["sst","uwind","vwind","vapor","cloud","rain"], "The variables for building the model.")

#reanalysis dataset experiment
flags.DEFINE_string("reanalysis_dir", "./data/reanalysis_dataset", "Path to save the origin data.")
flags.DEFINE_string("reanalysis_npz_dir", "./data/reanalysis_dataset/final", "Path to save trained models.")

# ---------- Training ----------
flags.DEFINE_string("encoder_save_dir", "./model/AutoEncoder/model_storage", "Path to save the encodered data and models.")
flags.DEFINE_float("train_eval_split", 0.1, "Percentage amount of testing data to use for eval.")
flags.DEFINE_integer("random_seed", 10, "Seed to use for random number generation and shuffling.")
flags.DEFINE_integer("latent_dim", 64, "Number of the latent features.")
flags.DEFINE_integer("num_epochs", 20, "Number of epochs to train for.")
flags.DEFINE_integer("batch_size", 4, "The batch size for training.")
flags.DEFINE_integer("sequence_length", 6, "Sequence lenghth for predicting.")

params = flags.FLAGS
params(sys.argv)
