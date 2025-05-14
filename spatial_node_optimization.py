import numpy as np
from numpy import random
import random
import matplotlib.pyplot as plt


import pandas as pd
from timeit import default_timer as timer
from sklearn.metrics import accuracy_score

#Choose data for analysis

''' 
Data was inspired by video: https://www.youtube.com/watch?v=LLBGiAAZqAM and 
was taken from here: https://gist.github.com/armgilles/194bcff35001e7eb53a2a8b441e8b2c6
'''

#Real Data
data=pd.read_csv("pokemon.csv").rename(columns={"Type 1": "Type"})

data = data.query("Type.isin(('Electric', 'Grass'))")

#X_data = data[[ 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed',  ]] # Features
X_data = data[[ 'HP', 'Attack', 'Defense', 'Speed',  ]] # Features
Y_data = (data['Type'] == 'Electric') 

#Some more transformation for work with nodes allowance 
x=X_data.to_numpy()
x=np.transpose(x)



y=Y_data.to_numpy()
y=np.expand_dims(y, axis=1)
y=np.transpose(y)
y=np.squeeze(y,axis=0)

#Saving data for further manipulations
residual=x

#Model construction

#Find row
k=len(x[:,0]) #amount of rows

#Creat training/test sets
train_split=int(0.8*np.size(data, axis=0))
x_train,y_train=x[:,:train_split],y[:train_split]
x_test,y_test=x[:,train_split:],y[train_split:]

residual=x_train


#Create node classes with different functionalities


class Add_a_function ():
    
    def __init__(self, coordinates, a=0.001):
        
        self.a=a
        self.coordinates=coordinates
    
    def _params(self):
        
        return self.a
    
    def forward (self, x):
        
        return x+self.a
    
    def gen_with_params (self, x, params):
        
        return x+params


class Multiply_a_function ():
    
    def __init__(self, coordinates, a=1.001):
        
        self.a=a
        self.coordinates=coordinates
    
    def _params(self):
        
        return self.a
    
    def forward (self, x):
        
        return x*self.a
    
    def gen_with_params (self, x, params):
        
        return x*params

    
class Scale_a_function ():
    
    def __init__(self, coordinates, a=1.1):
         
        self.a=a
        self.coordinates=coordinates
    
    def _params(self):
        
        return self.a
    
    def forward (self, x):
        
        return np.sign(x)*(np.abs(x))**self.a
    
    def gen_with_params (self, x, params):
        
        return np.sign(x)*(np.abs(x))**params   


class Sin_function ():
    
    def __init__(self, coordinates, a=1):
         
        self.a=a
        self.coordinates=coordinates
    
    def _params(self):
        
        return self.a
    
    def forward (self, x):
        
        return np.sin(self.a*x)
    
    def gen_with_params (self, x, params):
        
        return np.sin(params*x)


class Variable_add_function():
    
    #we add a variable from data to previous computings
    
    def __init__(self, coordinates):
        
        self.raw_number=random.choice(list(i for i in range(k))) #choice among given rows of initial data
        self.coordinates=coordinates
    
    def _params(self):
        
        return self.raw_number
    
    def forward (self, x):
        
        return x+residual[self.raw_number]
    
    def gen_with_params (self, x, params):
        
        return x+residual[params]


class Variable_multiply_function():
    
    #we add a variable from data to previous computings
    
    def __init__(self, coordinates):
        
        self.raw_number=random.choice(list(i for i in range(k))) #choice among given rows of initial data
        self.coordinates=coordinates
    
    def _params(self):
        
        return self.raw_number
    
    def forward (self, x):
        
        return x*residual[self.raw_number]
    
    def gen_with_params (self, x, params):
        
        return x*residual[params]

#Fancy node to show the power of current method

class Linear_regression ():
    
    def __init__(self, coordinates, y):
        
        self.y=y
        self.a=0
        self.b=0
        self.coordinates=coordinates
    
    def _params(self):
        
        n = np.size(x)

        
        x_average = np.mean(x)
        y_average = np.mean(self.y)

        
        cov = np.sum(self.y*x) - n*x_average*y_average
        dev = np.sum(x*x) - n*x_average*y_average

        # calculating regression coefficients
        self.b = cov/dev
        self.a = y_average - self.b*x_average
        
        return (self.a,self.b)
    
    def forward (self, x):

        return self.a+x*self.b
    
    def gen_with_params (self, x, params):
        
        return params[0]+params[1]*x


#Adding sin(x) to the previously obtained data
class Add_Sin_function ():
    
    def __init__(self, coordinates, a=1):
        
         
        self.a=a
        self.coordinates=coordinates
    
    def _params(self):
        
        return (self.a)
    
    def forward (self, x):
        
        return x+np.sin(self.a*x)
    
    def gen_with_params (self, x, params):
        
        return x+np.sin(params*x)       

class Multiply_Sin_function ():
    
    def __init__(self, coordinates, a=1):
        
        #self.a=random.randint(-10,10) 
        self.a=a
        self.coordinates=coordinates
    
    def _params(self):
        
        return (self.a)
    
    def forward (self, x):
        
        return x*np.sin(self.a*x)
    
    def gen_with_params (self, x, params):
        
        return x*np.sin(params*x)


class Variable_add_sin_function():
    
    #we add a variable from data to previous computings
    
    def __init__(self, coordinates):
        
        self.raw_number=random.choice(list(i for i in range(k))) #choice among given rows of initial data
        self.coordinates=coordinates
    
    def _params(self):
        
        return self.raw_number
    
    def forward (self, x):
        
        return x+np.sin(residual[self.raw_number])
    
    def gen_with_params (self, x, params):
        
        return x+np.sin(residual[params])


#Make instances for nodes 
node_1=Add_a_function((1,1))
node_2=Add_a_function((2,1),a=-0.001)
node_3=Multiply_a_function((2,1))
node_4=Multiply_a_function((2,2),a=0.999)
node_5=Scale_a_function((0,1))
node_6=Scale_a_function((1,0),a=0.9)
node_7=Sin_function((3,3), a=2)
node_8=Variable_add_function((-1,-1))
node_9=Variable_multiply_function((-1,0))
node_10=Linear_regression((3,4),y=y)





#Create string class. Main class for optimization

class String():
    
    def __init__(self, coordinates, logits, UF_initial):
        self.coordinates=coordinates
        self.logits=logits
        self.UF_string=[]
        self.Node_list_string=[]
        self.param_node_list=[]
        
        #Some inyternalization of utility function, doesn't matter now. Later it probably will has sense
        self.UF_max=UF_initial
        self.UF=UF_initial
    
    def append_UF(self):
        self.UF_string.append(self.UF)
    
    def append_Node_string(self, node):
        self.Node_list_string.append(node)
    
    def append_param_list (self, param):
        self.param_node_list.append(param)
        
    def pass_through_nodes(self, x):
        
        for i in range(len(self.Node_list_string)):
            func=self.Node_list_string[i]
            logits=func.gen_with_params(x,self.param_node_list[i])
            x=logits
        
        return logits



#Generate the next node function
def generate_line (current_coord, list_of_nodes: list):
    
    list_of_probs=[]
    list_of_distances=[]
    
    #compute distances for each node
    for i in range(len(list_of_nodes)):
        
        list_of_quasi_dist=[]
        for j in range(len(list_of_nodes[i].coordinates)):
            dist=(current_coord[j]-list_of_nodes[i].coordinates[j])
            distance=dist*dist
            list_of_quasi_dist.append(distance)
        
        big_distance=sum(list_of_quasi_dist)
        list_of_distances.append(big_distance)
    
    #compute quasi-probabilities from inverses of distances
    for k in range(len(list_of_distances)):
        
        #eliminating 100% probability of beign in the current node
        if list_of_distances[k]==0:
            inv_dist=0
        else:
            inv_dist=1/(list_of_distances[k])
            
        inv_total_dist=1/sum(list_of_distances)
        prob=inv_dist/inv_total_dist
        list_of_probs.append(prob)
    
    chosen_node=random.choices(list_of_nodes, weights=list_of_probs,k=1)[0]
    
    #Generating next node
    return chosen_node



    

    
#MSE Function of Joy. Utility function can be anything in general

def mse_joy (y, y_pred):
    
    mse_j = -1*(np.mean((y - y_pred)**2))
    
    return mse_j

#Hyperparametertes
EPOCH=1000

#Initial data to analyse
x_init=x_train[0]


#Instance for string



#UF_initial=mse_joy(y=y_train, y_pred=x_init) #initial meaning of utility function for later comparison


#Function to make binary choice

#Sigmoid function
def sigmoid (logits_to_remake):
    
    logits_to_remake=np.array(logits_to_remake, dtype=np.float64)
    
    return 1/(1+np.exp(-logits_to_remake))


#Above average labeling
def higher_than_mean(logits_to_binary):
    
    mean=np.mean(logits_to_binary)
    binary_ar=[]
    
    for item in logits_to_binary:
        if item>=mean:
            binary=1
        else:
            binary=0
        binary_ar.append(binary)
    
    binary_arr=np.array(binary_ar, dtype=float)
    
    return binary_arr



#Simple list for accuracy beside utility function, but they are the same
acc_evolution=[]

#UF_initial=mse_joy(y=y_train, y_pred=x_init)

label_init=higher_than_mean(logits_to_binary=x_init)
#label_init=np.round(sigmoid(logits_to_remake=x_init))
accuracy=accuracy_score(y_true=y_train, y_pred=label_init)
UF_initial=accuracy_score(y_true=y_train, y_pred=label_init)
acc_evolution.append(accuracy)

alpha=1 #parametere for loosing task for the systeme 
UF_max=UF_initial
string=String((0,0), x_init,UF_initial=UF_initial)
string.append_UF()
print(string.UF_string[-1])

# #Super fancy node in its own dimension, not used now
# class Segment_node():
    
#     def __init__(self, coordinates, epochs, residual):
        
#         self.coordinates=coordinates
#         self.epochs=epochs
#         self.residual=residual
        
#         self.internal_string=String(coordinates=(0,0),logits=residual[0],UF_initial=UF_max)
        
#         #Other node instances. Could be set during initializing
#         self.internal_node_1=Add_a_function((1,1))
#         self.internal_node_2=Add_a_function((2,1),a=-0.001)
#         self.internal_node_3=Multiply_a_function((2,1))
#         self.internal_node_4=Multiply_a_function((2,2),a=0.999)
#         self.internal_node_5=Scale_a_function((0,1))
#         self.internal_node_6=Scale_a_function((1,0),a=0.9)
#         self.internal_node_7=Sin_function((3,3), a=2)
#         self.internal_node_8=Variable_add_function((-1,-1))
#         self.internal_node_9=Variable_multiply_function((-1,0))
        
#         self.list_of_segmented_nodes=[self.internal_node_1,self.internal_node_2,self.internal_node_3,self.internal_node_4,
#                                       self.internal_node_5, self.internal_node_6, self.internal_node_7, self.internal_node_8,
#                                       self.internal_node_9]
        
#     def _params(self):
        
#         return (self.internal_string.Node_list_string, self.internal_string.param_node_list)
    
    
#     def forward(self, x):
        
#         local_UF_max=self.internal_string.UF_max
#         #global UF_max
        
#         x_internal=self.residual[0]
        
#         for epoch in range(self.epochs):
    
#             node=generate_line(self.internal_string.coordinates,list_of_nodes=self.list_of_segmented_nodes) #attraction-like choice 
            
#             #print(node)
            
#             #string.coordinates=node.coordinates
            
#             internal_logits=node.forward(x_internal)
#             logits=x*internal_logits #applying given function
            
#             #print(logits)
            
#             UF=mse_joy(y=y,y_pred=logits) #utility function
#             self.internal_string.UF=UF
#             #print(UF)
            
#             if UF>=local_UF_max*alpha:
#                 print('Oooohhhuuu!')
#                 self.internal_string.append_UF()
#                 x_internal=internal_logits #in successful case computations are used for further function applying
#                 self.internal_string.append_Node_string(node=node)
#                 self.internal_string.append_param_list(node._params())
                
#                 #Update coordinates of the string ? May be in general case is better. Should check
#                 self.internal_string.coordinates=node.coordinates
                
#                 #Update maximum of the utility function
#                 if UF>local_UF_max:
#                     local_UF_max=UF
#                     self.internal_string.UF_max=local_UF_max
#                     print(local_UF_max)
                
#                 #preparings for further applying of the same function
#                 UFtry=[] 
#                 UFtry.append(-1e30)
                
#                 UFtry_count=0 #need for skipping very big while loops in case of very little gradual improvements of utility function
                
#                 #if utility function is still improving
#                 while UF>=UF_max*alpha and UF>UFtry[-1] and UFtry_count<=50:
                    
                    
#                     internal_logits=node.forward(x)
                    
#                     logits=x*internal_logits
#                     UF=mse_joy(y=y,y_pred=logits)
#                     self.internal_string.UF=UF
                    
                    
#                     if UF>local_UF_max*alpha:
                        
#                         UFtry.append(self.internal_string.UF_string[-1])
#                         UFtry_count+=1
                        
#                         self.internal_string.append_UF()
#                         x_internal=internal_logits
#                         self.internal_string.append_Node_string(node=node)
#                         self.internal_string.append_param_list(node._params())
                        
#                     else:
#                         print("Noooo!")
#                         UFtry.append(1e10)
        
#         return logits
    
#     def gen_with_params (self, x, params):
        
#         internal_x=self.residual[0]
        
#         for i in range(len(params[0])):
#             func=params[0][i]
#             #print(func)
#             #print(params[1][i])
#             logits=func.gen_with_params(x=internal_x, params=params[1][i])
#             internal_x=logits
        
#         return x*internal_x


       

# node_11=Segment_node(coordinates=(0,-1),epochs=100, residual=residual)

# node_12=Segment_node(coordinates=(-3,-1),epochs=100, residual=residual)

node_13=Multiply_Sin_function(coordinates=(-2,-1))
node_14=Add_Sin_function(coordinates=(-3,-3))

node_15=Variable_add_sin_function(coordinates=(-2,3))

#List of playing nodes


list_of_current_nodes=[node_1,node_2,node_3,node_4,node_7,node_13,node_14, node_15, node_8, node_9]




#Searching loop
x=x_init

start=timer()

for epoch in range(EPOCH):
    

    node=generate_line(string.coordinates,list_of_nodes=list_of_current_nodes) #attraction-like choice 
    
    #print(node)
    
    #string.coordinates=node.coordinates
    
    logits=node.forward(x) #applying given function
    
    #print(logits)
    
    #UF=mse_joy(y=y_train,y_pred=logits) #utility function
    
    labels_pred=higher_than_mean(logits_to_binary=logits)
    #labels_pred=np.round(sigmoid(logits_to_remake=logits))
    accuracy=accuracy_score(y_true=y_train, y_pred=labels_pred)
    UF=accuracy_score(y_true=y_train, y_pred=labels_pred)
    string.UF=UF
    #print(UF)
    
    if string.UF>=string.UF_max*alpha:
        print('Oooohhhuuu!')
        string.append_UF()
        acc_evolution.append(accuracy) #update list of accuracies
        x=logits #in successful case computations are used for further function applying
        string.append_Node_string(node=node)
        string.append_param_list(node._params())
        
        #Update coordinates of the string ? May be in general case is better. Should check
        string.coordinates=node.coordinates
        
        #Update maximum of the utility function
        if string.UF>=string.UF_max*alpha:
            UF_max=UF
            string.UF_max=UF_max
            #print(UF_max)
        
        #preparings for further applying of the same function
        UFtry=[] 
        UFtry.append(-1e30)
        
        UFtry_count=0 #need for skipping very big while loops in case of very little gradual improvements of utility function
        
        #if utility function is still improving
        while UF>=UF_max*alpha and UF>UFtry[-1] and UFtry_count<=50:
            
            
            logits=node.forward(x)
            
            
            #UF=mse_joy(y=y_train,y_pred=logits)
            
            labels_pred=higher_than_mean(logits_to_binary=logits)
            #labels_pred=np.round(sigmoid(logits_to_remake=logits))
            UF=accuracy_score(y_true=y_train, y_pred=labels_pred)
            accuracy=accuracy_score(y_true=y_train, y_pred=labels_pred)
            string.UF=UF
            
            
            if string.UF>=string.UF_max*alpha:
                
                UFtry.append(string.UF_string[-1])
                UFtry_count+=1
                
                string.append_UF()
                acc_evolution.append(accuracy) #update accuracy list
                x=logits
                string.append_Node_string(node=node)
                string.append_param_list(node._params())
                
            else:
                print("Noooo!")
                UFtry.append(1e10)

#some information during training   
print(string.UF_string[:5], string.UF_string[-5:])
#print(string.Node_list_string[:5], string.Node_list_string[-5:])
#print(string.Node_list_string[:5], string.param_node_list[-5:])
print(acc_evolution[:5],acc_evolution[-5:])







#Main results
logits_for_max=string.pass_through_nodes(x=x_init)
#UF_for_max=mse_joy(y=y_train,y_pred=logits_for_max)


labels_pred_max=higher_than_mean(logits_to_binary=logits_for_max)
UF_for_max=accuracy_score(y_true=y_train, y_pred=labels_pred_max)
#labels_pred_max=np.round(sigmoid(logits_to_remake=logits_for_max))
accuracy_for_max=accuracy_score(y_true=y_train, y_pred=labels_pred_max)
print(f'UF for the train data is {UF_for_max}')
print(f'accuracy for the train data is {accuracy_for_max}')

end=timer()

print(f'time needed: {end-start}')


#Plot Utility Function
plt.plot(string.UF_string)
plt.show()



#Preparing for labels in test period
mean_for_max_train=np.mean(logits_for_max)

#Generating method for the model. Could be functionalised

x_init=x_test[0]

residual=x_test



x=x_init

#we apply function by function to our data
y_model=string.pass_through_nodes(x=x)
#best_UF=mse_joy(y=y_test,y_pred=y_model)

labels_list=[]

for item in y_model:
    
    if item>mean_for_max_train:
        local_label=1
    else:
        local_label=0
    labels_list.append(local_label)

labels_model=np.array(labels_list, dtype=float)

best_UF=accuracy_score(y_true=y_test, y_pred=labels_model)
#labels_model=np.round(sigmoid(logits_to_remake=y_model))
best_accuracy=accuracy_score(y_true=y_test, y_pred=labels_model)
print(f'UF for the test data is {best_UF}')
print(f'accuracy for the test data is {best_accuracy}')



#plt.plot(y_model, label='Model')
plt.plot(labels_model, label='Model')
plt.plot(y_test, label='Real Data')
plt.legend()
plt.show()


