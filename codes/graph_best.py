import csv
import matplotlib.pyplot as plt

r = csv.reader(open('results.csv','rb'),delimiter='\t')

min_lin,no_lin = 10,0
min_sigm,no_sigm = 10,0
min_tanh,no_tanh = 10,0
min_rbfl1,no_rbfl1 = 10,0
min_rbfl2,no_rbfl2 = 10,0
min_rbflinf,no_rbflinf = 10,0

neurons = [5,10,15,20,25,30,35,40,45,50]

lst_lin = []
lst_sigm = []
lst_tanh = []
lst_rbfl1 = []
lst_rbfl2 = []
lst_rbflinf = []

for row in r:
    break

for row in r:
    if row[2] == 'NMAE':
        if row[0] == 'lin':
            lst_lin.append(float(row[8]))
            if min_lin > float(row[8]):
                min_lin = float(row[8])
                no_lin = row[1]
        if row[0] == 'sigm':
            lst_sigm.append(float(row[8]))
            if min_sigm > float(row[8]):
                min_sigm = float(row[8])
                no_sigm = row[1]
        if row[0] == 'tanh':
            lst_tanh.append(float(row[8]))
            if min_tanh > float(row[8]):
                min_tanh = float(row[8])
                no_tanh = row[1]
        if row[0] == 'rbf_l1':
            lst_rbfl1.append(float(row[8]))
            if min_rbfl1 > float(row[8]):
                min_rbfl1 = float(row[8])
                no_rbfl1 = row[1]
        if row[0] == 'rbf_l2':
            lst_rbfl2.append(float(row[8]))
            if min_rbfl2 > float(row[8]):
                min_rbfl2 = float(row[8])
                no_rbfl2 = row[1]
        if row[0] == 'rbf_linf':
            lst_rbflinf.append(float(row[8]))
            if min_rbflinf > float(row[8]):
                min_rbflinf = float(row[8])
                no_rbflinf = row[1]

plt.plot(neurons, lst_lin)
plt.plot(neurons, lst_sigm)
plt.plot(neurons, lst_tanh)
plt.plot(neurons, lst_rbfl1)
plt.plot(neurons, lst_rbfl2)
plt.plot(neurons, lst_rbflinf)
plt.xlabel('Number of neurons')
plt.ylabel('NMAE')
plt.title('User cold start')
plt.legend(['linear', 'sigmoid', 'tanh', 'rbf_l1', 'rbf_l2', 'rbf_linf'], loc='upper right')
plt.savefig('users.png')

print 'For user'

r = csv.reader(open('results.csv','rb'),delimiter='\t')


for row in r:
    if row[0] == 'lin' and row[1] == no_lin:
        print 'lin',row[8],no_lin 
    if row[0] == 'sigm' and row[1] == no_sigm:
        print 'sigm',row[8],no_sigm 
    if row[0] == 'tanh' and row[1] == no_tanh:
        print 'tanh',row[8],no_tanh 
    if row[0] == 'rbf_l1' and row[1] == no_rbfl1:
        print 'rbf_l1',row[8],no_rbfl1  
    if row[0] == 'rbf_l2' and row[1] == no_rbfl2:
        print 'rbf_l2',row[8],no_rbfl2 
    if row[0] == 'rbf_linf' and row[1] == no_rbflinf:
        print 'rbf_linf',row[8],no_rbflinf 

print 
print 

r = csv.reader(open('results.csv','rb'),delimiter='\t')

min_lin,no_lin = 10,0
min_sigm,no_sigm = 10,0
min_tanh,no_tanh = 10,0
min_rbfl1,no_rbfl1 = 10,0
min_rbfl2,no_rbfl1 = 10,0
min_rbflinf,no_rbflinf = 10,0

lst_lin = []
lst_sigm = []
lst_tanh = []
lst_rbfl1 = []
lst_rbfl2 = []
lst_rbflinf = []

for row in r:
    if row[2] == 'NMAE':
        if row[0] == 'lin':
            lst_lin.append(float(row[14]))
            if min_lin > float(row[14]):
                min_lin = float(row[14])
                no_lin = row[1]
        if row[0] == 'sigm':
            lst_sigm.append(float(row[14]))
            if min_sigm > float(row[14]):
                min_sigm = float(row[14])
                no_sigm = row[1]
        if row[0] == 'tanh':
            lst_tanh.append(float(row[14]))
            if min_tanh > float(row[14]):
                min_tanh = float(row[14])
                no_tanh = row[1]
        if row[0] == 'rbf_l1':
            lst_rbfl1.append(float(row[14]))
            if min_rbfl1 > float(row[14]):
                min_rbfl1 = float(row[14])
                no_rbfl1 = row[1]
        if row[0] == 'rbf_l2':
            lst_rbfl2.append(float(row[14]))
            if min_rbfl2 > float(row[14]):
                min_rbfl2 = float(row[14])
                no_rbfl2 = row[1]
        if row[0] == 'rbf_linf':
            lst_rbflinf.append(float(row[14]))
            if min_rbflinf > float(row[14]):
                min_rbflinf = float(row[14])
                no_rbflinf = row[1]

plt.plot(neurons, lst_lin)
plt.plot(neurons, lst_sigm)
plt.plot(neurons, lst_tanh)
plt.plot(neurons, lst_rbfl1)
plt.plot(neurons, lst_rbfl2)
plt.plot(neurons, lst_rbflinf)
plt.xlabel('Number of neurons')
plt.ylabel('NMAE')
plt.title('Item cold start')
plt.legend(['linear', 'sigmoid', 'tanh', 'rbf_l1', 'rbf_l2', 'rbf_linf'], loc='upper right')
plt.savefig('items.png')

print 'For item'


r = csv.reader(open('results.csv','rb'),delimiter='\t')

for row in r:
    if row[0] == 'lin' and row[1] == no_lin:
        print 'lin',row[14],no_lin 
    if row[0] == 'sigm' and row[1] == no_sigm:
        print 'sigm',row[14],no_sigm 
    if row[0] == 'tanh' and row[1] == no_tanh:
        print 'tanh',row[14],no_tanh 
    if row[0] == 'rbf_l1' and row[1] == no_rbfl1:
        print 'rbf_l1',row[14],no_rbfl1  
    if row[0] == 'rbf_l2' and row[1] == no_rbfl2:
        print 'rbf_l2',row[14],no_rbfl2 
    if row[0] == 'rbf_linf' and row[1] == no_rbflinf:
        print 'rbf_linf',row[14],no_rbflinf 