import numpy as np
import scipy.io as scio
from sklearn.model_selection import StratifiedShuffleSplit

path = './mvdata/GBMCensorData.mat'
train_size = 0.7
def cal_split(name):
    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, train_size=0.7, random_state=10)
    _data = scio.loadmat('GBMCensorData.mat')
    mRNA = _data['mRNA']
    miRNA = _data['miRNA']
    DNA = _data['DNA']
    time = _data['time']
    censor = _data['censor']
    label = _data['label']


    for train_idx, test_idx in ss.split(mRNA, label):
        mRNA_train = mRNA[train_idx]
        mRNA_test = mRNA[test_idx]
        miRNA_train = miRNA[train_idx]
        miRNA_test = miRNA[test_idx]
        DNA_train = DNA[train_idx]
        DNA_test = DNA[test_idx]
        time_train = time[train_idx]
        time_test = time[test_idx]
        censor_train = censor[train_idx]
        censor_test = censor[test_idx]


    # scio.savemat('E:/GCGCN_Data/BIC_Test_idx.mat', {'BIC_Test_idx': (test_idx+1)})
    scio.savemat('GBM_train.mat',
                 mdict={'mRNA': mRNA_train, 'miRNA': miRNA_train, 'DNA': DNA_train, 'time': time_train, 'censor': censor_train})
    scio.savemat('GBM_test.mat',
                 mdict={'mRNA': mRNA_test, 'miRNA': miRNA_test, 'DNA': DNA_test, 'time': time_test, 'censor': censor_test})

if __name__ == '__main__':
    cal_split('GBM')
