import pickle
import numpy as np
import sys
sys.path.append("/mnt/g81linux/gurobi810/linux64/lib") 
import gurobipy as grby
from gurobipy import *
import numpy as np
from random import randint 
import pandas as pd

tl_split0 = np.load('../amelia_data/bert_small/train/true_label_split0.npy') # true label
pl_split0 = np.load('../amelia_data/bert_small/train/pred_label_split0.npy')
var = np.load('../amelia_data/bert_small/train/var_split0.npy')
uc_split0 = np.sqrt(var)
mp_split0 = np.load('../amelia_data/bert_small/train/mean_prob_split0.npy')
me_split0 = np.load('../amelia_data/bert_small/train/entropy_split0.npy')

tl_escalation = np.load('../amelia_data/bert_small/train/true_label_escalation.npy') # true label
pl_escalation = np.load('../amelia_data/bert_small/train/pred_label_escalation.npy')
var = np.load('../amelia_data/bert_small/train/var_escalation.npy')
uc_escalation = np.sqrt(var)
mp_escalation = np.load('../amelia_data/bert_small/train/mean_prob_escalation.npy')
me_escalation = np.load('../amelia_data/bert_small/train/entropy_escalation.npy')

tl_negative = np.load('../amelia_data/bert_small/train/true_label_negative.npy') # true label
pl_negative = np.load('../amelia_data/bert_small/train/pred_label_negative.npy')
me_negative = np.load('../amelia_data/bert_small/train/entropy_negative.npy')

tl_negative = tl_negative[:5000]
pl_negative = pl_negative[:5000]
me_negative = me_negative[:5000]

tl = np.concatenate((tl_split0, tl_escalation, tl_negative), axis=0)
pl = np.concatenate((pl_split0, pl_escalation, pl_negative), axis=0)
me = np.concatenate((me_split0, me_escalation, me_negative), axis=0)

tl[tl==382] = 381
pl[pl==382] = 381


# construct the optimization parameters
l= np.zeros((len(tl),382))
E = me

for loop in range(len(tl_split0)):
    l[loop,int(tl[loop])]=1
    l[loop,381]=0.5
    
# escalation and negative data set have ones on the last column
for loop in range(len(tl_split0), len(tl)):
    l[loop, 381]=0.5    
    

try:
    m = grby.Model("Optimal Threshold Problem Entropy Only")
    
    rowcount = len(tl)
    ncl = 382
    
    x = m.addVars(rowcount*ncl, vtype=GRB.BINARY, name="x")
    d = m.addVar(lb=0.0, ub=3.0, vtype=GRB.CONTINUOUS, name="d")
    
    
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
        for v in m.getVars():
            if ((v.VarName.find('d') != -1)):
                print('%s %g' % (v.varName, v.x))
        print('Obj: %g' % m.objVal)
        
    
    print('')
except GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))