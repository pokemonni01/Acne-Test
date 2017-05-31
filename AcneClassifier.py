import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
import weka.core.serialization as serialization
from weka.core.dataset import Instances, Instance
import os




jvm.start()
# Set Path Data Set
# data_dir = "/Users/wachirapong/PycharmProjects/Acne Test/"
data_dir = os.getcwd()+"/"

def loadModel():
    return Classifier(jobject=serialization.read(data_dir + "Test.model"))

def loadCSVDataSet():
    loader = Loader(classname="weka.core.converters.CSVLoader")
    dataset = loader.load_file(data_dir + "black-and-white.csv")
    dataset.class_is_last()
    return dataset

def buildMultilayerPerceptronClassifier(dataset):
    cls = Classifier(classname="weka.classifiers.functions.MultilayerPerceptron")
    cls.options = ["-L", "0.3", "-M", "0.2", "-N", "500", "-V", "0", "-S", "0", "-E", "20", "-H", "a"]
    cls.build_classifier(dataset)
    return cls

def classifyDataset(classifier, dataset):
    for index, inst in enumerate(dataset):
        pred = classifier.classify_instance(inst)
        dist = classifier.distribution_for_instance(inst)
        print(str(index + 1) + ": label index=" + str(pred) + ", class distribution=" + str(dist))

# Load CSV Data Set
datas = loadCSVDataSet()
classifier = buildMultilayerPerceptronClassifier(datas)

# evaluation = Evaluation(datas)
# evaluation.crossvalidate_model(classifier, datas, 10, Random(42))
# print(evaluation.summary())
# print("pctCorrect: " + str(evaluation.percent_correct))
# print("incorrect: " + str(evaluation.incorrect))

data2 = datas.get_instance(1)
data2.set_value(0, 478)
data2.set_value(1, 118)
data2.set_value(2, 928)
data2.set_value(3, 151)
data2.set_value(4, 1257)
data2.set_value(5, 220)
pred = classifier.classify_instance(data2)
print ( str(pred) )


