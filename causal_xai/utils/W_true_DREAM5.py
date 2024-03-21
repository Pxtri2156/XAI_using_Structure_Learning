import numpy as np 
import pandas as pd 


def load_network(net):
    filename = f'/dataset/DREAM5/gold_standard_edges_only/DREAM5_NetworkInference_Edges_Network{net}.tsv'
    df = pd.read_csv(filename, header=None, sep='[ \t]', engine='python')
    df[0] = df[0].apply(lambda x: x.replace('g','').replace('G','')).astype(int)
    df[1] = df[1].apply(lambda x: x.replace('g','').replace('G','')).astype(int)
    df[2] = df[2].astype(float) # imoprtant for later to check for equality
    return df

def read_dream5(net, **kwargs):
    ex_path = f'/dataset/DREAM5/net{net}/net{net}_expression_data.tsv'
    df = pd.read_csv(ex_path, sep='\t').dropna()
    n_samples = df.shape[0]
    n_features = df.shape[1]
    cols = df.columns.tolist()[:n_features]
    X = df.values
    dimension = len(cols)
    features = cols
    return X, dimension, features

def buildB_true(net, dimension): 
    net_df = load_network(net)
    B_true = np.zeros((dimension, dimension))
    for i, row in net_df.iterrows(): 
        if row[2] == 1: 
            B_true[int(row[0])-1, int(row[1])-1] = 1
    return B_true
def main(): 
    net = '1'
    saving_path = '/dataset/DREAM5/X_B_true/'
    X, d, features = read_dream5(net)
    B_true = buildB_true(net, d)
    print(X.shape)
    print(B_true.shape)
    np.savetxt(saving_path + f'X_{net}.csv', X, delimiter=',')
    np.savetxt(saving_path + f'B_true_{net}.csv', B_true, delimiter=',')
    return 

if __name__ == '__main__':
    main()