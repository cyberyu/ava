import pickle
import numpy as np
import sys
sys.path.append("/mnt/g81linux/gurobi810/linux64/lib") 
import gurobipy as grby
from gurobipy import *
import numpy as np
import random
from random import randint 
import pandas as pd
from sklearn.metrics import accuracy_score
from scipy.stats import entropy

global CLASS_NO

CLASS_NO=150


v_20_train = pickle.load(open('../public_data/train_90percent.pkl','rb'))
#v_20_test = pickle.load(open('/mnt/amelia_data/381classes/10percent_test.pkl','rb'))
v_20_negative = pickle.load(open('../public_data/negative_90percent.pkl','rb'))


df_train = pd.read_csv('../public_data/train.tsv', delimiter='\t', header=None)
df_train.columns = ['Questions','numeric label']
df_train['numeric label'] = df_train['numeric label'].astype(np.int64)
tlabel = df_train['numeric label'].values

df_test = pd.read_csv('../public_data/test.tsv', delimiter='\t', header=None)
df_test.columns = ['Questions','numeric label']
df_test['numeric label'] = df_test['numeric label'].astype(np.int64)
telabels = df_test['numeric label'].values


#tlabel = pickle.load(open('/mnt/amelia_data/381classes/test_labels.pkl','rb'))

test_mean = np.mean(v_20_train,1)
test_variance = np.std(v_20_train,1) 






# calcualte entropy using test_mean
entropy([1/2, 1/2], base=2)


negative_mean = np.mean(v_20_negative,1)
negative_variance = np.std(v_20_negative,1)

num_test = 500  #randomly sample a number of irrelevant questions
index_test = random.sample(range(len(tlabel)), num_test)

tlabel = tlabel[index_test]
test_mean = test_mean[index_test]
test_variance = test_variance[index_test,:]

test_entropy = np.zeros(len(test_mean))

for i in range(len(test_mean)):
    test_entropy[i]=entropy(test_mean[i,:], base=2)


num_negative = 100

index_negative= random.sample(range(len(negative_mean)), num_negative)
negative_variance=negative_variance[index_negative,:]
negative_mean=negative_mean[index_negative,:]

negative_entropy = np.zeros(len(negative_mean))

for i in range(len(negative_mean)):
    negative_entropy[i]=entropy(negative_mean[i,:], base=2)
    

# alllabels= np.concatenate((tlabel, np.ones(len(negative_mean))*CLASS_NO))

# mp = np.concatenate((test_mean, negative_mean), axis=0)
# uc = np.concatenate((test_variance, negative_variance), axis=0)

me = np.concatenate((test_entropy, negative_entropy), axis=0)
tl = np.concatenate((tlabel, np.ones(len(negative_mean))*CLASS_NO))

# construct the optimization parameters
l= np.zeros((len(tl),CLASS_NO+1))
E = me

for loop in range(len(tlabel)):
    l[loop,int(tl[loop])]=1
    l[loop,CLASS_NO]=0.5
    
# escalation and negative data set have ones on the last column
for loop in range(len(tlabel), len(tl)):
    l[loop, CLASS_NO]=0.5    
    

try:
    m = grby.Model("Optimal Threshold Problem Entropy Only")
    
    rowcount = len(tl)
    ncl = CLASS_NO+1
    
    x = m.addVars(rowcount*ncl, vtype=GRB.BINARY, name="x")
    d = m.addVar(lb=0.0, ub=10.0, vtype=GRB.CONTINUOUS, name="d")
    
    
    for loop in range(rowcount):
        # adding entropy constraint
        m.addGenConstrIndicator(x[ncl*(loop+1)-1], 1, d, GRB.LESS_EQUAL, E[loop]-0.0001)
        m.addGenConstrIndicator(x[ncl*(loop+1)-1], 0, d, GRB.GREATER_EQUAL, E[loop])

    for loop in range(rowcount):
        m.addConstr(quicksum(x[j] for j in range(loop*ncl,(loop+1)*ncl)),GRB.EQUAL, 1)
            
    m.setObjective(quicksum(x[i]*x[i] for i in range(rowcount*ncl))-quicksum(2*x[i]*l[int(i/ncl),i%ncl] for i in range(rowcount*ncl)), GRB.MINIMIZE)

    m.optimize()

    if m.status == GRB.INFEASIBLE:
        #cul_err=cul_err + 0.5
        #inf_cnt=inf_cnt+1
        #cul_err = cul_err+sum(A[rowind,]*l[rowind,])
        print('infeasible')

    else:   
        print('Obj: %g' % m.objVal)
        for v in m.getVars():
            if (v.VarName.find('d')!=-1):
                d_optimal = v.x
                print('%s %g' % (v.varName, v.x))
                
        # reconstruct the solutions
        x_recon= np.zeros((rowcount,ncl))
        x_recon_label = np.zeros(rowcount)
        
        
        # x start from x[0]
        for v in m.getVars():
            if (v.VarName.find('x')!=-1):
                if (v.x!=0):
                    x_no = int(v.VarName[1:].replace('[','').replace(']',''))
                    x_recon[int(x_no/ncl), x_no%ncl]=1
#                 print('%s %g' % (v.varName, v.x))
        
    
        for row in range(len(x_recon)):
            for col in range(ncl):
                if x_recon[row,col]==1:
                    x_recon_label[row] = col
            
        
        print(np.min(x_recon_label))
        print(np.max(x_recon_label))

        print('Training Accuracy Score is '+ str(accuracy_score(tl, x_recon_label)))
        print ('In Training, Number of samples predicited as irrelevant ' + str(np.count_nonzero(x_recon_label==CLASS_NO)))
        print ('In Training, Number of samples are truely irrelevant ' + str(np.count_nonzero(tl==CLASS_NO)))
    
    print('')
except GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))