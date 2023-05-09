#params for this model
import sys

from absl import flags

# ---------- Pre-Processing ----------
flags.DEFINE_string("preprocess_out_dir", "./data/records", "Path to save the pre-process output.")
flags.DEFINE_integer("shrink_size", 10, "Size for shrinking the origin data.")

# remote sensing dataset experiment
flags.DEFINE_string("remote_sensing_dataset_dir", "./data/remote_sensing_dataset", "Path to save the origin data.")
flags.DEFINE_string("remote_sensing_npz_dir", "./data/remote_sensing_dataset/final", "Path to save trained models.")
flags.DEFINE_list("remote_sensing_variables", ["sst","uwind","vwind","vapor","cloud","rain"], "The variables for building the model.")

# ---------- Training ----------
flags.DEFINE_string("encoder_save_dir", "./model/AutoEncoder/model_storage", "Path to save the encodered data or models.")
flags.DEFINE_float("train_eval_split", 0.1, "Percentage amount of testing data to use for eval.")
flags.DEFINE_integer("random_seed", 10, "Seed to use for random number generation and shuffling.")
flags.DEFINE_integer("latent_dim", 64, "Number of the latent features.")
flags.DEFINE_integer("num_epochs", 128, "Number of epochs to train for.")
flags.DEFINE_integer("batch_size", 2, "The batch size for training.")

params = flags.FLAGS
params(sys.argv)
