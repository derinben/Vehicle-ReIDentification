import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import zipfile

# to mount google drive
from google.colab import drive 
drive.mount("/content/drive", force_remount=True)

local_zip = '/content/drive/MyDrive/Datasets/VeRi.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/Veri')
zip_ref.close()

class data():
    
    def __init__(self, train_path , test_path):
        self.train_path = train_path
        self.test_path = test_path
        
    def load_data(self): #to load and rename columns
        train_df = pd.read_excel(self.train_path)
        test_df = pd.read_excel(self.test_path)
        
        train_df.columns = ['ImageName' , 'VehicleID' ,'CameraID' , 'ColorID' ,'TypeID']
        test_df.columns = ['ImageName' , 'VehicleID' ,'CameraID' , 'ColorID' ,'TypeID']        

        return train_df, test_df
    
    def Enumerate(self, col): #to maintain sequence in the categories from 0 to (number of unique values-1)
      
        dict_train = {}
        dict_test  = {}
        self.train_df, self.test_df = self.load_data()
  
        arr_train = np.sort(self.train_df[col].unique())
        arr_test = np.sort(self.test_df[col].unique())

        for count, item in enumerate(arr_train):
            dict_train[item] = count
        for count, item in enumerate(arr_test):
            dict_test[item] = count          

        return dict_train ,dict_test
    
    def change(self):

        VehicleID_dict_train, VehicleID_dict_test = self.Enumerate('VehicleID')
        ColorID_dict_train, ColorID_dict_test = self.Enumerate('ColorID')
        TypeID_dict_train, TypeID_dict_test = self.Enumerate('TypeID')
        
        self.train_df = self.train_df.replace({"VehicleID": VehicleID_dict_train})
        self.train_df = self.train_df.replace({"ColorID": ColorID_dict_train})
        self.train_df = self.train_df.replace({"TypeID": TypeID_dict_train})

        self.test_df = self.test_df.replace({"VehicleID": VehicleID_dict_test})
        self.test_df = self.test_df.replace({"ColorID": ColorID_dict_test})
        self.test_df = self.test_df.replace({"TypeID": TypeID_dict_test})

        return self.train_df, self.test_df
      
      
train_path = 'data/train_label.xlsx'
test_path = 'data/test_label.xlsx'

datafiles = data(train_path, test_path)
train_df , test_df = datafiles.change()


#to find the active cameras for every VehicleID

grpdf = train_df.groupby(['CameraID','VehicleID']).count()
table = pd.pivot_table(grpdf, values='ImageName', index=['VehicleID'], columns=['CameraID'], aggfunc=np.sum)
train_df['CameraID'] =  train_df['CameraID'].astype(str).str[-2:].astype(np.int64)  
table = table.reset_index()
#print(table.columns)

#Positive pairs

def create_positive_pairs():
    pairlist=[]  
    uniVID = train_df['VehicleID'].value_counts().index.sort_values() #gets unique values of VehicleID
    for VID in uniVID:
        x = np.squeeze(table[table['VehicleID'] ==VID].values)[1:]  #to query 'table' row by row using VID
        idx = pd.Index(x).notnull() 
        image_ID_list = []
        for i in range(len(idx)):
            if idx[i] == True:
                image_ID_list.append(i+1)
                i+1
            else:
                i+1        
        tdf = []
        
        for ele in image_ID_list:
            tdf.append(train_df['ImageName'][(train_df['CameraID']==ele) & (train_df['VehicleID'] == VID)].values)
        pairs = []
        for i in range(0,2):
            ind1 = random.randint(0, len(tdf[0+i])-1)
            x1 = tdf[0+i][ind1]
            pairs.append(x1)
        pairlist.append(pairs)
     
    return pairlist
  

pairlist1  = create_positive_pairs()
pairlist2  = create_positive_pairs()
pairlist3  = create_positive_pairs()
pairlist4  = create_positive_pairs()
pairlist5  = create_positive_pairs()

ar_p1 = np.array(pairlist1)
ar_p2 = np.array(pairlist2)
ar_p3 = np.array(pairlist3)
ar_p4 = np.array(pairlist4)
ar_p5 = np.array(pairlist5)

ar_pairlists = np.vstack((ar_p1,ar_p2,ar_p3,ar_p4,ar_p5))

positive_pair_df = pd.DataFrame(data = ar_pairlists, columns = ['Image1','Image2'])
positive_pair_df['label'] = 1
positive_pair_df.drop_duplicates(inplace=True)

# print(positive_pair_df.shape)
# print(positive_pair_df.head())

def create_neg_pairs():
    pairs = []
    uniVID = train_df['VehicleID'].value_counts().index.sort_values() #gets unique values of VehicleID
    for VID in uniVID:
        x = np.squeeze(table[table['VehicleID'] ==VID].values)[1:]  #to query 'table' row by row using VID
        idx = pd.Index(x).notnull() 
        image_ID_list = []
        for i in range(len(idx)):
            if idx[i] == True:
                image_ID_list.append(i+1)
                i+1
            else:
                i+1     
        #print("For Vehicle ID {}, the active cameras are {}, total = {} ".format(VID,image_ID_list, len(image_ID_list)))
        
        tdf = []
        i = 0 
        for ele in image_ID_list:
            tdf.append(train_df['ImageName'][(train_df['CameraID']==ele) & (train_df['VehicleID'] == VID)].values)  
            i=i+1

        ind1 = random.randint(0,len(image_ID_list)-1)
        ind2 = random.randint(0, len(tdf[ind1])-1)
        x1 = tdf[ind1][ind2]
        randomVID = [el for el in uniVID if el != VID]
        random_vid = random.choice(randomVID)
        rdf = train_df['ImageName'][train_df['VehicleID'] == random_vid]
        x2 = random.choice(rdf.to_list())
        pairs.append([x1,x2])

    return pairs
    

pairlist1  = create_neg_pairs()
pairlist2  = create_neg_pairs()
pairlist3  = create_neg_pairs()
pairlist4 = create_neg_pairs()
pairlist5 = create_neg_pairs()

ar_p1 = np.array(pairlist1)
ar_p2 = np.array(pairlist2)
ar_p3 = np.array(pairlist3)
ar_p4 = np.array(pairlist4)
ar_p5 = np.array(pairlist5)

ar_pairlists = np.vstack((ar_p1,ar_p2,ar_p3,ar_p4,ar_p5))
negative_pair_df = pd.DataFrame(data = ar_pairlists,columns =['Image1','Image2'])
negative_pair_df['label'] = 0
negative_pair_df.drop_duplicates(inplace=True)

# print(negative_pair_df.shape)
# print(negative_pair_df.head(10))

siamese_df = positive_pair_df.copy()
siamese_df = siamese_df.append(negative_pair_df,ignore_index=True)
siamese_df = siamese_df.sample(frac=1).reset_index(drop=True)
siamese_df['label'] = siamese_df['label'].astype(str)
#print(siamese_df.head())
