import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from train_model import test_model
from keras.models import load_model
import pickle

# Parameters
# to_load = "nietzsche"
to_load = "obama2"
length = 200

outfile = open("generated/{}_long.txt".format(to_load), "w")
model = load_model("saved_models/{}.h5".format(to_load))
char_to_indices = pickle.load(open("saved_models/{}c2i.p".format(to_load), "rb"))
indices_to_char = pickle.load(open("saved_models/{}i2c.p".format(to_load), "rb"))

for temperature in [0.2, 0.4, 0.6]:
	generated_string = test_model(model=model, char_to_indices=char_to_indices, indices_to_char=indices_to_char, \
	 seed_string=" ", temperature=0.3, test_length=length)
	output = "Temperature: {} Generated string: {}".format(temperature, generated_string)
	print(output)
	outfile.write(output + "\n")
	outfile.flush()
outfile.close()