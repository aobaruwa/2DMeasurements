from landmark import utils
import pickle
import sys

fname=sys.argv[1]

data=pickle.load(open(fname,'rb'))

pc=list(data.values())[0]['PC']
lm=list(data.values())[0]['LM']

utils.plot_landmark(pc,lm)
