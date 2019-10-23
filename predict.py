import argparse
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time


from keras.models import load_model
from PIL import Image
from numpy import asarray

######

parser = argparse.ArgumentParser()
parser.add_argument("images", nargs="+", metavar="image")
parser.add_argument("-w", "--weights", metavar="weight-file", default="weights.h5")
args = parser.parse_args()

print("Loading model")
model = load_model(args.weights)

start = time.time()
SIZE=256

for img in args.images:
    image = Image.open(img)
    img_resized = image.resize((SIZE,SIZE))
    data = asarray(img_resized)/255.0 # rgb 0 - 255 to 0 - 1
    y = model.predict(data.reshape(1, SIZE, SIZE, 3))  # data reshape for a single image
    print(img, ": ", y)

end = time.time();
print("Predicted {0:d} images in {1:3.2f}s".format(len(args.images), end - start))
