import os
import pickle
import string
import random

# for loading/processing the images
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

# models
from keras.applications.vgg16 import VGG16
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# for everything else

import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import make_grid
from torchvision.transforms import Resize

###########################################################
#
###########################################################
def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

###########################################################
#
###########################################################
def extract_features(file, model):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224,224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img)
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3)
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True, verbose = None)
    return features

###########################################################
#
###########################################################
class Images:
    #######################################################
    def __init__(self, model):
        self.model = model
        self.features = {}

    #######################################################
    def LoadData(self, filename):
        if os.path.isfile(filename):
            print("Loading features file")
            self.data = pickle.load(open(filename, 'rb'))
        else:
            print("Features file not found, creating a new one")

    #######################################################
    def SaveData(self, filename):
        print("Saving Feature file")
        pickle.dump(self.data, open(filename, 'wb'))

    #######################################################
    def LoadDir(self, path):
        print("Load dir "+path)
        error_found = False
        self.rootpath = path
        nb_new_images = 0
        nb_cur_images = 0
        for (root, dirs, files) in os.walk(path):
            root = root.replace('\\', '/')
            lastdir = root.split("/")[-1]
            if lastdir.startswith("_ignore_"):
                continue
            for filename in files:
                print(f"root={root} Filename={filename}")
                if filename in self.data.keys():
                    image = self.data[filename]
                    # check if image already found
                    if image.new_path :
                        if image.new_path == root:
                            nb_cur_images += 1
                        else:
                            if os.path.isfile(image.new_path+"/"+filename):
                                print("Error : file "+filename+" found twice in")
                                print("  - "+image.new_path)
                                print("  - "+root)
                                os.rename(image.new_path+"/"+filename, image.new_path+"/"+get_random_string(6)+filename)
                                error_found = True
                            else:
                                del self.data[filename]
                    else: # image without newpath, error
                        print("Error in loaded data, no newpath in image "+filename)
                else:
                    if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg") or filename.lower().endswith(".png"):
                        nb_new_images += 1
                        image = Image(root, filename)
                        self.data[filename] = image
        if not error_found:
            print("Loaded OK. Data contains "+str(nb_cur_images)+", added "+str(nb_new_images)+" new images")
        else:
            print("Error while loading")
        return not error_found
    
    #######################################################
    def GetFeatures(self):
        i = 0
        for filename in self.data.keys():
            image = self.data[filename]
            if image.features is None:
                i+=1
                if (i % 100) == 0:
                    print("Save temp data -------------------------------")
                    self.SaveData(self.rootpath+"/"+"tmpdata.pkl")
                image.GetFeatures(model)

    #######################################################
    def Clustering(self, n_labels):
        # Create the array of filenames and features
        filenames_list = []
        features_list = []

        for key in self.data.keys():
            filenames_list.append(key)
            if self.data[key].features_ok :
                features_list.append(self.data[key].features)

        features_array = np.array(features_list)
        print(features_array.shape)

        # reshape so feature_array is n times 4096 vectors
        features_array = features_array.reshape(-1,4096)
        print(features_array.shape)

        # reduce the amount of dimensions in the feature vector
        print("reduce the amount of dimensions in the feature vector")
        pca = PCA(n_components=n_labels, random_state=22)
        pca.fit(features_array)
        x = pca.transform(features_array)

        print("cluster feature vectors")
        # cluster feature vectors
        kmeans = KMeans(n_clusters=n_labels, random_state=22, n_init = "auto")
        kmeans.fit(x)

        # holds the cluster id and the images { id: [images] }
        self.groups = {}
        for file, cluster in zip(filenames_list,kmeans.labels_):
            if cluster not in self.groups.keys():
                self.groups[cluster] = []
                self.groups[cluster].append(file)
            else:
                self.groups[cluster].append(file)

    #######################################################
    def GroupImages_dir(self):

        for cluster in self.groups.keys():
            # create cluster directory
            cluster_dir = self.rootpath+"/"+str(cluster)
            if not os.path.isdir(cluster_dir):
                os.mkdir(cluster_dir)
            # move files 
            for filename in self.groups[cluster]:
                image = self.data[filename]
                if not image.new_path == cluster_dir:
                    src = image.new_path + "/" + image.filename
                    dst = cluster_dir + "/" + image.filename
                    if not src == dst:
                        try:
                            os.rename(src, dst)
                            image.new_path = cluster_dir
                        except:
                            dst = self.rootpath + "/" + image.filename
                            os.rename(src, dst)
                            image.new_path = self.rootpath

            #print(str(cluster) + ": "+str(groups[cluster]))

        print("Group images by directory OK")

    #######################################################
    def GroupImages_name(self):

        cluster_index = 0
        for cluster in self.groups.keys():
            for filename in self.groups[cluster]:
                image = self.data[filename]
                if filename.startswith("#"):
                    print(filename.split("#"))
                    root_filename = filename.split("#")[2]
                    newfilename = f"#{cluster_index:03}#{root_filename}"
                else:
                    newfilename = f"#{cluster_index:03}#{filename}"
                #print(f"filename newfilename} / image {image.new_path}:{image.filename}")
                src = image.new_path + "/" + filename
                dst = image.new_path + "/" + newfilename
                os.rename(src, dst)
                image.new_path = self.rootpath
            cluster_index += 1

            #print(str(cluster) + ": "+str(groups[cluster]))

        print("Group images by filename OK")


###########################################################
#
###########################################################
class Image:
    #######################################################
    def __init__(self, path, filename):
        self.original_path = path
        self.filename = filename
        self.new_path = path
        self.moved_path = None
        self.features = None
        self.features_ok = False

    #######################################################
    def GetFeatures(self, model):
        print("loading "+self.new_path+"/"+self.filename)
        # load the image as a 224x224 array
        try:
            img = load_img(self.new_path+"/"+self.filename, target_size=(224,224))
            # convert from 'PIL.Image.Image' to numpy array
            img = np.array(img)
            # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
            reshaped_img = img.reshape(1,224,224,3)
            # prepare image for model
            imgx = preprocess_input(reshaped_img)
            # get the feature vector
            self.features = model.predict(imgx, use_multiprocessing=True, verbose = None)
            self.features_ok = True
        except:
            print("Loading error")
        

###############################################################################
###############################################################################
####
#### usage : 
#### python clustering_images.py dir n
#### where dir contains the image to sort and n is the number of clusters
####

if len(sys.argv) >= 1:
    img_path = sys.argv[1]

feat_datafile = "features.pkl"

model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

images = Images(model)

images.LoadData(img_path+"\\"+feat_datafile)
# load new images
if not images.LoadDir(img_path):
    print("Error found")
    quit()

images.GetFeatures()
images.SaveData(img_path+"\\"+feat_datafile)
nb_clusters = int(len(images.data)/2+1)
images.Clustering(nb_clusters)
images.GroupImages_name() # rename files according to cluster numbner
# images.GroupImages_dir() # move files to directores according to cluster numbner
images.SaveData(img_path+"\\"+feat_datafile)


