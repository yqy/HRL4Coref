import argparse

DATA_DIT="/Users/yqy/data/coreference/annotation_chinese/"
#parse arguments
parser = argparse.ArgumentParser(description="Experiemts for Coreference Resolution (by qyyin)\n")

parser.add_argument("-train_data",default = DATA_DIT + "data_raw/train",help="specify train data file")
parser.add_argument("-train_gold",default = DATA_DIT + "gold/train",help="specify train data file with gold chains")
parser.add_argument("-dev_data",default = DATA_DIT + "data_raw/dev",help="specify dev data file")
parser.add_argument("-dev_gold",default = DATA_DIT + "gold/dev",help="specify dev data file with gold chains")
parser.add_argument("-test_data",default = DATA_DIT + "data_raw/test",help="specify test data file")
parser.add_argument("-test_gold",default = DATA_DIT + "gold/test",help="specify test data file with gold chains")

parser.add_argument("-language",default = "cn",help="specify language")
parser.add_argument("-embedding",default="/Users/yqy/data/coreference/embedding/embedding.",help="embedding dir")

#parameters for neural network
parser.add_argument("-echos",default=10,type=int,help="Echo Times")
parser.add_argument("-lr",default=0.03,type=float,help="Learning Rate")
parser.add_argument("-batch",default=15,type=int,help="batch size")
parser.add_argument("-dev_prob",default=0.1,type=float,help="probability of development set")
parser.add_argument("-dropout_prob",default=0.5,type=float,help="probability of dropout")
parser.add_argument("-random_seed",default=110,type=int,help="random seed")

args = parser.parse_args()
