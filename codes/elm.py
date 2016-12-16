import csv
from scipy.io import loadmat
from hpelm import ELM
import numpy as np 
from sklearn.metrics import mean_absolute_error as mae 
from sklearn.metrics import mean_squared_error as mse

w = csv.writer(open('results.csv','wb'),delimiter = '\t')

w.writerow(["Neuron type", "Number of Nodes", "Error Metric", 
        "user split 1", "user split 2", "user split 3", "user split 4",
        "user split 5", "user average", "item split 1", "item split 1",
        "item split 2", "item split 3", "item split 4", "item split 5",
        "item average"])

neurons = [5,10,15,20,25,30,35,40,45,50]
node_type = ['lin','sigm','tanh','rbf_l1','rbf_l2','rbf_linf']

for node in node_type:
    print "\n",node
    for neuron in neurons:
        print "\n",neuron
        print "ITEM\n"
        #rating_read = csv.reader(open('../dataset/ml-100k/u1.base','rb'), delimiter='\t')
        rating_mat = loadmat('../dataset/item/folds_item_cold_start.mat')
        train_ids_read = open('../dataset/item/train_index.txt','rb')
        test_ids_read = open('../dataset/item/test_index.txt','rb')
        user_read = csv.reader(open('../dataset/ml-100k/u.user.mod','rb'), delimiter='|')
        item_read = csv.reader(open('../dataset/ml-100k/u.item','rb'), delimiter='|')
        #test_read = csv.reader(open('../dataset/ml-100k/u1.test','rb'), delimiter='\t')


        """
        ratings = []
        T = []
        for row in rating_read:
            ratings.append([int(row[0]),int(row[1]),int(row[2])])
            temp = [0] * 5
            temp[int(row[2])-1] = 1
            T.append(temp)
        """


        users = {}
        for row in user_read:
            if int(row[0]) not in users:
                users[int(row[0])] = [int(x) for x in row[1:]]

        items = {}
        for row in item_read:
            if int(row[0]) not in items:
                items[int(row[0])] = [int(x) for x in row[5:]]

        """
        X = []
        for rating in ratings:
            X.append(users[rating[0]] + items[rating[1]])

        test = []
        test_rat = []
        for row in test_read:
            test.append(users[int(row[0])] + items[int(row[1])])
            temp = [0] * 5
            temp[int(row[2])-1] = 1
            test_rat.append(temp)
        """

        #################################SPLIT 1#####################################
        train_ids = map(int, train_ids_read.readline().strip().split(','))
        test_ids = map(int, test_ids_read.readline().strip().split(','))

        X = []
        T = []
        mat_iid = 0
        for _id in train_ids:
            if _id != 0:
                for i in range(1,944):
                    if rating_mat['train_1'].T[mat_iid][i-1] != 0:
                        X.append(items[_id] + users[i])
                        temp = [0] * 5
                        temp[rating_mat['train_1'].T[mat_iid][i-1] - 1] = 1
                        T.append(temp)
                mat_iid += 1

        test = []
        test_rat = []
        mat_iid = 0
        for _id in test_ids:
            if _id != 0:
                for i in range(1,944):
                    if rating_mat['test_1'].T[mat_iid][i-1] != 0:
                        test.append(items[_id] + users[i])
                        temp = [0] * 5
                        temp[rating_mat['test_1'].T[mat_iid][i-1] - 1] = 1
                        test_rat.append(temp)
                        #print mat_iid
                mat_iid += 1


        X = np.asarray(X, dtype=np.uint8)
        T = np.asarray(T, dtype=np.uint8)
        test = np.asarray(test, dtype=np.uint8)
        test_rat = np.asarray(test_rat, dtype=np.uint8)

        ##print X.shape,test.shape

        elm = ELM(X.shape[1], T.shape[1]) 
        elm.add_neurons(neuron, node)
        elm.train(X,T, "LOO")
        Y = elm.predict(test)
        

        pred = np.argmax(Y, axis=1)
        true = np.argmax(test_rat, axis=1)

        print 'Split 1 RMSE: ', mse(true,pred)**0.5
        print 'Split 1 NMAE: ', mae(true,pred)/4 

        i1_rmse = mse(true,pred)**0.5
        i1_nmae = mae(true,pred)/4
        #################################SPLIT 2#####################################
        train_ids = map(int, train_ids_read.readline().strip().split(','))
        test_ids = map(int, test_ids_read.readline().strip().split(','))

        X = []
        T = []
        mat_iid = 0
        for _id in train_ids:
            if _id != 0:
                for i in range(1,944):
                    if rating_mat['train_2'].T[mat_iid][i-1] != 0:
                        X.append(items[_id] + users[i])
                        temp = [0] * 5
                        temp[rating_mat['train_2'].T[mat_iid][i-1] - 1] = 1
                        T.append(temp)
                mat_iid += 1

        test = []
        test_rat = []
        mat_iid = 0
        for _id in test_ids:
            if _id != 0:
                for i in range(1,944):
                    if rating_mat['test_2'].T[mat_iid][i-1] != 0:
                        test.append(items[_id] + users[i])
                        temp = [0] * 5
                        temp[rating_mat['test_2'].T[mat_iid][i-1] - 1] = 1
                        test_rat.append(temp)
                        #print mat_iid
                mat_iid += 1


        X = np.asarray(X, dtype=np.uint8)
        T = np.asarray(T, dtype=np.uint8)
        test = np.asarray(test, dtype=np.uint8)
        test_rat = np.asarray(test_rat, dtype=np.uint8)

        #print X.shape,test.shape

        elm = ELM(X.shape[1], T.shape[1]) 
        elm.add_neurons(neuron, node)
        elm.train(X,T, "LOO")
        Y = elm.predict(test)

        pred = np.argmax(Y, axis=1)
        true = np.argmax(test_rat, axis=1)

        print 'Split 2 RMSE: ', mse(true,pred)**0.5
        print 'Split 2 NMAE: ', mae(true,pred)/4

        i2_rmse = mse(true,pred)**0.5
        i2_nmae = mae(true,pred)/4
        #################################SPLIT 3#####################################
        train_ids = map(int, train_ids_read.readline().strip().split(','))
        test_ids = map(int, test_ids_read.readline().strip().split(','))

        X = []
        T = []
        mat_iid = 0
        for _id in train_ids:
            if _id != 0:
                for i in range(1,944):
                    if rating_mat['train_3'].T[mat_iid][i-1] != 0:
                        X.append(items[_id] + users[i])
                        temp = [0] * 5
                        temp[rating_mat['train_3'].T[mat_iid][i-1] - 1] = 1
                        T.append(temp)
                mat_iid += 1

        test = []
        test_rat = []
        mat_iid = 0
        for _id in test_ids:
            if _id != 0:
                for i in range(1,944):
                    if rating_mat['test_3'].T[mat_iid][i-1] != 0:
                        test.append(items[_id] + users[i])
                        temp = [0] * 5
                        temp[rating_mat['test_3'].T[mat_iid][i-1] - 1] = 1
                        test_rat.append(temp)
                        #print mat_iid
                mat_iid += 1


        X = np.asarray(X, dtype=np.uint8)
        T = np.asarray(T, dtype=np.uint8)
        test = np.asarray(test, dtype=np.uint8)
        test_rat = np.asarray(test_rat, dtype=np.uint8)

        #print X.shape,test.shape

        elm = ELM(X.shape[1], T.shape[1]) 
        elm.add_neurons(neuron, node)
        elm.train(X,T, "LOO")
        Y = elm.predict(test)

        pred = np.argmax(Y, axis=1)
        true = np.argmax(test_rat, axis=1)

        print 'Split 3 RMSE: ', mse(true,pred)**0.5
        print 'Split 3 NMAE: ', mae(true,pred)/4

        i3_rmse = mse(true,pred)**0.5
        i3_nmae = mae(true,pred)/4
        #################################SPLIT 4#####################################
        train_ids = map(int, train_ids_read.readline().strip().split(','))
        test_ids = map(int, test_ids_read.readline().strip().split(','))

        X = []
        T = []
        mat_iid = 0
        for _id in train_ids:
            if _id != 0:
                for i in range(1,944):
                    if rating_mat['train_4'].T[mat_iid][i-1] != 0:
                        X.append(items[_id] + users[i])
                        temp = [0] * 5
                        temp[rating_mat['train_4'].T[mat_iid][i-1] - 1] = 1
                        T.append(temp)
                mat_iid += 1

        test = []
        test_rat = []
        mat_iid = 0
        for _id in test_ids:
            if _id != 0:
                for i in range(1,944):
                    if rating_mat['test_4'].T[mat_iid][i-1] != 0:
                        test.append(items[_id] + users[i])
                        temp = [0] * 5
                        temp[rating_mat['test_4'].T[mat_iid][i-1] - 1] = 1
                        test_rat.append(temp)
                        #print mat_iid
                mat_iid += 1


        X = np.asarray(X, dtype=np.uint8)
        T = np.asarray(T, dtype=np.uint8)
        test = np.asarray(test, dtype=np.uint8)
        test_rat = np.asarray(test_rat, dtype=np.uint8)

        #print X.shape,test.shape

        elm = ELM(X.shape[1], T.shape[1]) 
        elm.add_neurons(neuron, node)
        elm.train(X,T, "LOO")
        Y = elm.predict(test)

        pred = np.argmax(Y, axis=1)
        true = np.argmax(test_rat, axis=1)

        print 'Split 4 RMSE: ', mse(true,pred)**0.5
        print 'Split 4 NMAE: ', mae(true,pred)/4

        i4_rmse = mse(true,pred)**0.5
        i4_nmae = mae(true,pred)/4
        #################################SPLIT 5#####################################
        train_ids = map(int, train_ids_read.readline().strip().split(','))
        test_ids = map(int, test_ids_read.readline().strip().split(','))

        X = []
        T = []
        mat_iid = 0
        for _id in train_ids:
            if _id != 0:
                for i in range(1,944):
                    if rating_mat['train_5'].T[mat_iid][i-1] != 0:
                        X.append(items[_id] + users[i])
                        temp = [0] * 5
                        temp[rating_mat['train_5'].T[mat_iid][i-1] - 1] = 1
                        T.append(temp)
                mat_iid += 1

        test = []
        test_rat = []
        mat_iid = 0
        for _id in test_ids:
            if _id != 0:
                for i in range(1,944):
                    if rating_mat['test_5'].T[mat_iid][i-1] != 0:
                        test.append(items[_id] + users[i])
                        temp = [0] * 5
                        temp[rating_mat['test_5'].T[mat_iid][i-1] - 1] = 1
                        test_rat.append(temp)
                        #print mat_iid
                mat_iid += 1


        X = np.asarray(X, dtype=np.uint8)
        T = np.asarray(T, dtype=np.uint8)
        test = np.asarray(test, dtype=np.uint8)
        test_rat = np.asarray(test_rat, dtype=np.uint8)

        #print X.shape,test.shape

        elm = ELM(X.shape[1], T.shape[1]) 
        elm.add_neurons(neuron, node)
        elm.train(X,T, "LOO")
        Y = elm.predict(test)

        pred = np.argmax(Y, axis=1)
        true = np.argmax(test_rat, axis=1)

        print 'Split 5 RMSE: ', mse(true,pred)**0.5
        print 'Split 5 NMAE: ', mae(true,pred)/4

        i5_rmse = mse(true,pred)**0.5
        i5_nmae = mae(true,pred)/4

        print "\n\nUSER\n"
        #rating_read = csv.reader(open('../dataset/ml-100k/u1.base','rb'), delimiter='\t')
        rating_mat = loadmat('../dataset/user/folds_user_cold_start.mat')
        train_ids_read = open('../dataset/user/train_index.txt','rb')
        test_ids_read = open('../dataset/user/test_index.txt','rb')
        user_read = csv.reader(open('../dataset/ml-100k/u.user.mod','rb'), delimiter='|')
        item_read = csv.reader(open('../dataset/ml-100k/u.item','rb'), delimiter='|')
        #test_read = csv.reader(open('../dataset/ml-100k/u1.test','rb'), delimiter='\t')

        """
        ratings = []
        T = []
        for row in rating_read:
            ratings.append([int(row[0]),int(row[1]),int(row[2])])
            temp = [0] * 5
            temp[int(row[2])-1] = 1
            T.append(temp)
        """

        users = {}
        for row in user_read:
            if int(row[0]) not in users:
                users[int(row[0])] = [int(x) for x in row[1:]]

        items = {}
        for row in item_read:
            if int(row[0]) not in items:
                items[int(row[0])] = [int(x) for x in row[5:]]
        """
        X = []
        for rating in ratings:
            X.append(users[rating[0]] + items[rating[1]])

        test = []
        test_rat = []
        for row in test_read:
            test.append(users[int(row[0])] + items[int(row[1])])
            temp = [0] * 5
            temp[int(row[2])-1] = 1
            test_rat.append(temp)
        """

        #################################SPLIT 1#####################################
        train_ids = map(int, train_ids_read.readline().strip().split(','))
        test_ids = map(int, test_ids_read.readline().strip().split(','))

        X = []
        T = []
        mat_iid = 0
        for _id in train_ids:
            if _id != 0:
                for i in range(1,1683):
                    if rating_mat['train_1'][mat_iid][i-1] != 0:
                        X.append(users[_id] + items[i])
                        temp = [0] * 5
                        temp[rating_mat['train_1'][mat_iid][i-1] - 1] = 1
                        T.append(temp)
                mat_iid += 1

        test = []
        test_rat = []
        mat_iid = 0
        for _id in test_ids:
            if _id != 0:
                for i in range(1,1683):
                    if rating_mat['test_1'][mat_iid][i-1] != 0:
                        test.append(users[_id] + items[i])
                        temp = [0] * 5
                        temp[rating_mat['test_1'][mat_iid][i-1] - 1] = 1
                        test_rat.append(temp)
                        #print mat_iid
                mat_iid += 1


        X = np.asarray(X, dtype=np.uint8)
        T = np.asarray(T, dtype=np.uint8)
        test = np.asarray(test, dtype=np.uint8)
        test_rat = np.asarray(test_rat, dtype=np.uint8)

        ##print X.shape,test.shape

        elm = ELM(X.shape[1], T.shape[1]) 
        elm.add_neurons(neuron, node)
        elm.train(X,T, "LOO")
        Y = elm.predict(test)

        pred = np.argmax(Y, axis=1)
        true = np.argmax(test_rat, axis=1)

        print 'Split 1 RMSE: ', mse(true,pred)**0.5
        print 'Split 1 NMAE: ', mae(true,pred)/4

        u1_rmse = mse(true,pred)**0.5
        u1_nmae = mae(true,pred)/4
        #################################SPLIT 2#####################################
        train_ids = map(int, train_ids_read.readline().strip().split(','))
        test_ids = map(int, test_ids_read.readline().strip().split(','))

        X = []
        T = []
        mat_iid = 0
        for _id in train_ids:
            if _id != 0:
                for i in range(1,1683):
                    if rating_mat['train_2'][mat_iid][i-1] != 0:
                        X.append(users[_id] + items[i])
                        temp = [0] * 5
                        temp[rating_mat['train_2'][mat_iid][i-1] - 1] = 1
                        T.append(temp)
                mat_iid += 1

        test = []
        test_rat = []
        mat_iid = 0
        for _id in test_ids:
            if _id != 0:
                for i in range(1,1683):
                    if rating_mat['test_2'][mat_iid][i-1] != 0:
                        test.append(users[_id] + items[i])
                        temp = [0] * 5
                        temp[rating_mat['test_2'][mat_iid][i-1] - 1] = 1
                        test_rat.append(temp)
                        #print mat_iid
                mat_iid += 1


        X = np.asarray(X, dtype=np.uint8)
        T = np.asarray(T, dtype=np.uint8)
        test = np.asarray(test, dtype=np.uint8)
        test_rat = np.asarray(test_rat, dtype=np.uint8)

        #print X.shape,test.shape

        elm = ELM(X.shape[1], T.shape[1]) 
        elm.add_neurons(neuron, node)
        elm.train(X,T, "LOO")
        Y = elm.predict(test)

        pred = np.argmax(Y, axis=1)
        true = np.argmax(test_rat, axis=1)

        print 'Split 2 RMSE: ', mse(true,pred)**0.5
        print 'Split 2 NMAE: ', mae(true,pred)/4

        u2_rmse = mse(true,pred)**0.5
        u2_nmae = mae(true,pred)/4
        #################################SPLIT 3#####################################
        train_ids = map(int, train_ids_read.readline().strip().split(','))
        test_ids = map(int, test_ids_read.readline().strip().split(','))

        X = []
        T = []
        mat_iid = 0
        for _id in train_ids:
            if _id != 0:
                for i in range(1,1683):
                    if rating_mat['train_3'][mat_iid][i-1] != 0:
                        X.append(users[_id] + items[i])
                        temp = [0] * 5
                        temp[rating_mat['train_3'][mat_iid][i-1] - 1] = 1
                        T.append(temp)
                mat_iid += 1

        test = []
        test_rat = []
        mat_iid = 0
        for _id in test_ids:
            if _id != 0:
                for i in range(1,1683):
                    if rating_mat['test_3'][mat_iid][i-1] != 0:
                        test.append(users[_id] + items[i])
                        temp = [0] * 5
                        temp[rating_mat['test_3'][mat_iid][i-1] - 1] = 1
                        test_rat.append(temp)
                        #print mat_iid
                mat_iid += 1


        X = np.asarray(X, dtype=np.uint8)
        T = np.asarray(T, dtype=np.uint8)
        test = np.asarray(test, dtype=np.uint8)
        test_rat = np.asarray(test_rat, dtype=np.uint8)

        #print X.shape,test.shape

        elm = ELM(X.shape[1], T.shape[1]) 
        elm.add_neurons(neuron, node)
        elm.train(X,T, "LOO")
        Y = elm.predict(test)

        pred = np.argmax(Y, axis=1)
        true = np.argmax(test_rat, axis=1)

        print 'Split 3 RMSE: ', mse(true,pred)**0.5
        print 'Split 3 NMAE: ', mae(true,pred)/4

        u3_rmse = mse(true,pred)**0.5
        u3_nmae = mae(true,pred)/4
        #################################SPLIT 4#####################################
        train_ids = map(int, train_ids_read.readline().strip().split(','))
        test_ids = map(int, test_ids_read.readline().strip().split(','))

        X = []
        T = []
        mat_iid = 0
        for _id in train_ids:
            if _id != 0:
                for i in range(1,1683):
                    if rating_mat['train_4'][mat_iid][i-1] != 0:
                        X.append(users[_id] + items[i])
                        temp = [0] * 5
                        temp[rating_mat['train_4'][mat_iid][i-1] - 1] = 1
                        T.append(temp)
                mat_iid += 1

        test = []
        test_rat = []
        mat_iid = 0
        for _id in test_ids:
            if _id != 0:
                for i in range(1,1683):
                    if rating_mat['test_4'][mat_iid][i-1] != 0:
                        test.append(users[_id] + items[i])
                        temp = [0] * 5
                        temp[rating_mat['test_4'][mat_iid][i-1] - 1] = 1
                        test_rat.append(temp)
                        #print mat_iid
                mat_iid += 1


        X = np.asarray(X, dtype=np.uint8)
        T = np.asarray(T, dtype=np.uint8)
        test = np.asarray(test, dtype=np.uint8)
        test_rat = np.asarray(test_rat, dtype=np.uint8)

        #print X.shape,test.shape

        elm = ELM(X.shape[1], T.shape[1]) 
        elm.add_neurons(neuron, node)
        elm.train(X,T, "LOO")
        Y = elm.predict(test)

        pred = np.argmax(Y, axis=1)
        true = np.argmax(test_rat, axis=1)

        print 'Split 4 RMSE: ', mse(true,pred)**0.5
        print 'Split 4 NMAE: ', mae(true,pred)/4

        u4_rmse = mse(true,pred)**0.5
        u4_nmae = mae(true,pred)/4
        #################################SPLIT 5#####################################
        train_ids = map(int, train_ids_read.readline().strip().split(','))
        test_ids = map(int, test_ids_read.readline().strip().split(','))

        X = []
        T = []
        mat_iid = 0
        for _id in train_ids:
            if _id != 0:
                for i in range(1,1683):
                    if rating_mat['train_5'][mat_iid][i-1] != 0:
                        X.append(users[_id] + items[i])
                        temp = [0] * 5
                        temp[rating_mat['train_5'][mat_iid][i-1] - 1] = 1
                        T.append(temp)
                mat_iid += 1

        test = []
        test_rat = []
        mat_iid = 0
        for _id in test_ids:
            if _id != 0:
                for i in range(1,1683):
                    if rating_mat['test_5'][mat_iid][i-1] != 0:
                        test.append(users[_id] + items[i])
                        temp = [0] * 5
                        temp[rating_mat['test_5'][mat_iid][i-1] - 1] = 1
                        test_rat.append(temp)
                        #print mat_iid
                mat_iid += 1


        X = np.asarray(X, dtype=np.uint8)
        T = np.asarray(T, dtype=np.uint8)
        test = np.asarray(test, dtype=np.uint8)
        test_rat = np.asarray(test_rat, dtype=np.uint8)

        #print X.shape,test.shape

        elm = ELM(X.shape[1], T.shape[1]) 
        elm.add_neurons(neuron, node)
        elm.train(X,T, "LOO")
        Y = elm.predict(test)

        pred = np.argmax(Y, axis=1)
        true = np.argmax(test_rat, axis=1)

        print 'Split 5 RMSE: ', mse(true,pred)**0.5
        print 'Split 5 NMAE: ', mae(true,pred)/4

        u5_rmse = mse(true,pred)**0.5
        u5_nmae = mae(true,pred)/4

        w.writerow([node,neuron,'RMSE',u1_rmse,u2_rmse,u3_rmse,u4_rmse,u5_rmse,
            sum([u1_rmse,u2_rmse,u3_rmse,u4_rmse,u5_rmse])/5.0,
            i1_rmse,i2_rmse,i3_rmse,i4_rmse,i5_rmse, 
            sum([i1_rmse,i2_rmse,i3_rmse,i4_rmse,i5_rmse])/5.0])

        w.writerow([node,neuron,'NMAE',u1_nmae,u2_nmae,u3_nmae,u4_nmae,u5_nmae,
            sum([u1_nmae,u2_nmae,u3_nmae,u4_nmae,u5_nmae])/5.0,
            i1_nmae,i2_nmae,i3_nmae,i4_nmae,i5_nmae, 
            sum([i1_nmae,i2_nmae,i3_nmae,i4_nmae,i5_nmae])/5.0])