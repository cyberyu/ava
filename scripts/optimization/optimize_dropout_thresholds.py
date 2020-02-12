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


v_20_train = pickle.load(open('/mnt/amelia_data/381classes/10percent_train.pkl','rb'))
#v_20_test = pickle.load(open('/mnt/amelia_data/381classes/10percent_test.pkl','rb'))
v_20_negative = pickle.load(open('/mnt/amelia_data/381classes/10percent_negative.pkl','rb'))


df_train = pd.read_csv('/mnt/amelia_data/381classes/train.tsv', delimiter='\t', header=None)
df_train.columns = ['Questions','numeric label']
df_train['numeric label'] = df_train['numeric label'].astype(np.int64)
tlabel = df_train['numeric label'].values

df_test = pd.read_csv('/mnt/amelia_data/381classes/test.tsv', delimiter='\t', header=None)
df_test.columns = ['Questions','numeric label']
df_test['numeric label'] = df_test['numeric label'].astype(np.int64)
telabels = df_test['numeric label'].values


#tlabel = pickle.load(open('/mnt/amelia_data/381classes/test_labels.pkl','rb'))

test_mean = np.mean(v_20_train,1)
test_variance = np.std(v_20_train,1) 

negative_mean = np.mean(v_20_negative,1)
negative_variance = np.std(v_20_negative,1)

num_test = 1500  #randomly sample a number of irrelevant questions
index_test = random.sample(range(len(tlabel)), num_test)

tlabel = tlabel[index_test]
test_mean = test_mean[index_test]
test_variance = test_variance[index_test,:]

num_negative = 100

index_negative= random.sample(range(len(negative_mean)), num_negative)
negative_variance=negative_variance[index_negative,:]
negative_mean=negative_mean[index_negative,:]

alllabels= np.concatenate((tlabel, np.ones(len(negative_mean))*381))

mp = np.concatenate((test_mean, negative_mean), axis=0)
uc = np.concatenate((test_variance, negative_variance), axis=0)


# construct the optimization parameters
l= np.zeros((len(alllabels),382))*0.5
A= np.zeros((len(mp),382))
B= np.ones((len(mp),382))*0
# E = me


A[:,:-1]=mp
B[:,:-1]=uc

for loop in range(len(test_mean)):
    l[loop,int(alllabels[loop])]=1
    l[loop,381]=0.5

# escalation and negative data set have ones on the last column
for loop in range(len(test_mean), len(alllabels)):
    l[loop, 381]=1
    
#best result

try:
    m = grby.Model("Optimal Threshold Problem")
    
    rowcount = len(A)
    ncl = 382
    
#     x = m.addVars(rowcount*ncl, vtype=GRB.BINARY, name="x")
    
#     b = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="b")
#     c = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="c")
#     d = m.addVar(lb=0.0, ub=3.0, vtype=GRB.CONTINUOUS, name="d")
    
    x = m.addVars(rowcount*ncl, vtype=GRB.BINARY, name="x")
    y = m.addVars(rowcount*ncl, vtype=GRB.BINARY, name="y")
    z = m.addVars(rowcount*ncl, vtype=GRB.BINARY, name="z")
    b = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="b")
    c = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="c")
#     d = m.addVar(lb=0.0, ub=3.0, vtype=GRB.CONTINUOUS, name="d")
    
    
    for loop in range(rowcount):
        for class_ind in range(ncl-1):
            m.addGenConstrIndicator(y[loop*ncl+class_ind], 0, b, GRB.GREATER_EQUAL, A[loop,class_ind])   # b is the mean probability value  > A
            m.addGenConstrIndicator(y[loop*ncl+class_ind], 1, b, GRB.LESS_EQUAL, A[loop,class_ind])   # b is the mean probability value  < A
            m.addGenConstrIndicator(z[loop*ncl+class_ind], 0, c, GRB.LESS_EQUAL, B[loop,class_ind])      # c is the stand deviation < B
            m.addGenConstrIndicator(z[loop*ncl+class_ind], 1, c, GRB.GREATER_EQUAL, B[loop,class_ind])      # c is the stand deviation > B            
            m.addConstr(x[loop*ncl]==and_(z[loop*ncl],y[loop*ncl]))
        m.addConstr(quicksum(x[j] for j in range(loop*ncl,(loop+1)*ncl)),GRB.EQUAL, 1)
    m.setObjective(quicksum(x[i]*x[i] for i in range(rowcount*ncl))-quicksum(2*x[i]*l[int(i/ncl),i%ncl] for i in range(rowcount*ncl)), GRB.MINIMIZE)

    
#     for loop in range(rowcount):
#         for class_ind in range(ncl-1):
#             m.addGenConstrIndicator(x[loop*ncl+class_ind], 0, b, GRB.GREATER_EQUAL, A[loop,class_ind])   # b is the mean probability value,  if b > mp, 0
#             m.addGenConstrIndicator(x[loop*ncl+class_ind], 0, c, GRB.LESS_EQUAL, B[loop,class_ind])      # c is the stand deviation , if c < std, 0


#         m.addGenConstrIndicator(x[ncl*(loop+1)-1], 0, d, GRB.GREATER_EQUAL, E[loop]-0.0001)    # if   d > E[loop] - 0.0001,  x == 0, the escalation column should be 0
#         m.addGenConstrIndicator(x[ncl*(loop+1)-1], 1, d, GRB.LESS_EQUAL, E[loop])              # if   d <= E[loop],  x == 1, the escalation column need to be 1
    
#         m.addConstr(quicksum(x[j] for j in range(loop*ncl,(loop+1)*ncl)),GRB.EQUAL, 1)
                                  
#     m.setObjective(quicksum(x[i]*x[i] for i in range(rowcount*ncl))-quicksum(2*x[i]*l[int(i/ncl),i%ncl] for i in range(rowcount*ncl)), GRB.MINIMIZE)

    m.optimize()

    if m.status == GRB.INFEASIBLE:
        #cul_err=cul_err + 0.5
        #inf_cnt=inf_cnt+1
        #cul_err = cul_err+sum(A[rowind,]*l[rowind,])
        print('infeasible')

    else:   
        
        print('Obj: %g' % m.objVal)
        for v in m.getVars():
            if ((v.VarName.find('b') != -1) or (v.VarName.find('d') != -1) or (v.VarName.find('c') != -1) ):
                if (v.VarName.find('d') != -1):
                    d_optimal = v.x
                if (v.VarName.find('b') != -1):
                    b_optimal = v.x
                if (v.VarName.find('c') != -1):
                    c_optimal = v.x                
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

        print('Training Accuracy Score is '+ str(accuracy_score(alllabels, x_recon_label)))
        print ('In Training, Number of samples predicited as irrelevant ' + str(np.count_nonzero(x_recon_label==381)))
        print ('In Training, Number of samples are truely irrelevant ' + str(np.count_nonzero(alllabels==381)))
    
    print('')
except GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))