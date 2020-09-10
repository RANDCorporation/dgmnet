import torch
from torch_geometric.data import Data, InMemoryDataset, DataLoader
import pandas as pd
import numpy as np


def dataframe2onehot(df, node_attributes):
    """
    Convert the data in dataframe format to a torch one-hot encodded 
    format that the link prediction model can accept as input.
    
    The reason why this function receives as input both the dataframe of interest (df)
    and the NDSSL dataframe is that I want all zipcodes to be present in the feature
    matrix, even if df does not have every zipcode represented.
    """
    
    ## initialize and empty x tensor
    x = torch.empty(len(df), 0)
    
    if 'gender' in df.columns:
        ## one-hot encode gender
        gender_index = torch.LongTensor(df['gender'].values - 1).type(torch.int64).reshape((len(df), 1))
        gender_onehot = torch.LongTensor(len(df), 2)
        gender_onehot.zero_()
        gender_onehot = gender_onehot.scatter_(1, gender_index, 1).type(torch.float32);
        x = torch.cat((x, gender_onehot), dim=1)

    if 'worker' in df.columns:
        ## one-hot encode worker
        worker_index = torch.LongTensor(df['worker'].values - 1).type(torch.int64).reshape((len(df), 1))
        worker_onehot = torch.LongTensor(len(df), 2)
        worker_onehot.zero_()
        worker_onehot = worker_onehot.scatter_(1, worker_index, 1).type(torch.float32);
        x = torch.cat((x, worker_onehot), dim=1)

    if 'zipcode' in df.columns:
        ## map the 117 distinct zipcodes to the integers 0, ..., 116
        zipcode_original = df['zipcode'].values
        zipcode_dict = {i: j for j, i in enumerate(set(node_attributes['zipcode'].values))}
        zipcode_index = torch.LongTensor(np.asarray([zipcode_dict[i] for i in zipcode_original])).type(torch.int64).reshape((len(df), 1))

        ## one-hot encode zipcode
        zipcode_onehot = torch.LongTensor(len(df), len(zipcode_dict))
        zipcode_onehot.zero_()
        zipcode_onehot = zipcode_onehot.scatter_(1, zipcode_index, 1).type(torch.float32);
        x = torch.cat((x, zipcode_onehot), dim=1)

    if 'household_income' in df.columns:
        ## one-hot encode household income
        household_income_index = torch.LongTensor(df['household_income'].values - 1).type(torch.int64).reshape((len(df), 1))
        household_income_onehot = torch.LongTensor(len(df), 14)
        household_income_onehot.zero_()
        household_income_onehot = household_income_onehot.scatter_(1, household_income_index, 1).type(torch.float32);
        x = torch.cat((x, household_income_onehot), dim=1)

    if 'relationship' in df.columns:
        ## one-hot encode relationship
        relationship_index = torch.LongTensor(df['relationship'].values - 1).type(torch.int64).reshape((len(df), 1))
        relationship_onehot = torch.LongTensor(len(df), 4)
        relationship_onehot.zero_()
        relationship_onehot = relationship_onehot.scatter_(1, relationship_index, 1).type(torch.float32);
        x = torch.cat((x, relationship_onehot), dim=1)

    if 'age' in df.columns:
        age = torch.FloatTensor(df['age'].values).reshape(len(df), 1).type(torch.float32)
        x = torch.cat((x, age), dim=1)

    if 'household_size' in df.columns:
        household_size = torch.FloatTensor(df['household_size'].values).reshape(len(df), 1).type(torch.float32)
        x = torch.cat((x, household_size), dim=1)
    
    if 'household_workers' in df.columns:
        household_workers = torch.FloatTensor(df['household_workers'].values).reshape(len(df), 1).type(torch.float32)
        x = torch.cat((x, household_workers), dim=1)
    
    if 'household_vehicles' in df.columns:
        household_vehicles = torch.FloatTensor(df['household_vehicles'].values).reshape(len(df), 1).type(torch.float32)    
        x = torch.cat((x, household_vehicles), dim=1)
    
    return x

class NDSSLDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(NDSSLDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return ['edge_list.csv', 'edge_attributes.csv']

    @property
    def processed_file_names(self):
        return ['NDSSL_graph_full.pt']

    def process(self):
        data_list = []
        
        ## load the edge list
        edge_list = pd.read_csv(self.raw_paths[0], dtype=int) - 2000000 #the node id's start at 2000000, shift these 
        
        ## format the edge list
        target_nodes = edge_list.iloc[:,0].values
        source_nodes = edge_list.iloc[:,1].values
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.int64)
        
        ## load the (x,y) formatted data
        #x = torch.load(self.raw_paths[1], map_location=torch.device('cpu'))
        #y = torch.load(self.raw_paths[2], map_location=torch.device('cpu'))
        #train_mask = torch.load(self.raw_paths[3], map_location=torch.device('cpu')) == 1 
        #test_mask = torch.load(self.raw_paths[4], map_location=torch.device('cpu')) == 1 

        ## set the edge weights to be the duration (in hours)
        edge_attributes = pd.read_csv(self.raw_paths[1])['duration'].values/3600
        duration =  torch.FloatTensor(edge_attributes)
        ## previous approaches used the degree:
        #row, col = data.edge_index
        #data.edge_attr = (1. / degree(col, data.num_nodes)[col]).double()
        
        ## build the data
        #data = Data(edge_index=edge_index, x=x, y=y, train_mask=train_mask, test_mask=test_mask)
        data = Data(edge_index=edge_index)
        data.edge_weight = duration
        #data.train_mask = train_mask
        #data.test_mask = test_mask

        print(data.__dict__)
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])


class EgoDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(EgoDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['EGO_graph_second.csv', 'EGO_graph_second_attr.csv']
    @property
    def processed_file_names(self):
        return ['EGO_graph_second.pt']

    def download(self):
        pass
    
    def process(self):
        self.index = pd.read_csv(self.raw_paths[0]) - 1
        self.attrs = pd.read_csv(self.raw_paths[1])
        
        data_list = []
        
        self.attrs['Gender'] = le_Gender.fit_transform(self.attrs.iloc[:,0])
        self.attrs['Household.Id'] = le_HI.fit_transform(self.attrs.iloc[:,1])
        self.attrs['zipcode'] = le_zipcode.fit_transform(self.attrs.iloc[:,2])
        node_features = self.attrs.values
        
        node_features = torch.FloatTensor(node_features)
        #node_features = torch.LongTensor(node_features).unsqueeze(1)
        target_nodes = self.index.iloc[:,1].values
        source_nodes = self.index.iloc[:,0].values

        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        x = node_features

        # y = torch.FloatTensor([self.attrs.iloc[:,0].values])

        data = Data(x=x, edge_index=edge_index)
        data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class EgoDatasetWithAlters(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(EgoDatasetWithAlters, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['EGO_graph_third.csv', 'EGO_graph_third_attr.csv']
    @property
    def processed_file_names(self):
        return ['EGOwithAlters_third.pt']

    def download(self):
        pass
    
    def process(self):
        val_ratio = 0.05; test_ratio = 0.1
        self.index = pd.read_csv(self.raw_paths[0]) - 1
        self.attrs = pd.read_csv(self.raw_paths[1])
        
        num_nodes = np.max(self.index.values) + 1
        
        self.attrs['Gender'] = LabelEncoder().fit_transform(self.attrs.iloc[:,0])
        self.attrs['Household.Id'] = LabelEncoder().fit_transform(self.attrs.iloc[:,1])
        self.attrs['zipcode'] = LabelEncoder().fit_transform(self.attrs.iloc[:,2])
        
        data_list = []
        
        grouped = self.index.groupby('graph')
        for graph_id, group in grouped:
            attrs = self.attrs.loc[self.attrs.graph == graph_id,:].copy()
            node_features = attrs.iloc[:,:3].values

            node_features = torch.FloatTensor(node_features)
            # node_features = torch.LongTensor(node_features).unsqueeze(1)
            target_nodes = group.iloc[:,1].values
            source_nodes = group.iloc[:,0].values

            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            x = node_features

            # y = torch.FloatTensor([self.attrs.iloc[:,0].values])

            data = Data(x=x, edge_index=edge_index)
            data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
