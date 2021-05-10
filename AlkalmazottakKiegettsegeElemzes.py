# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
plt.rcParams.update({'figure.max_open_warning': 0})

#%% Adatgyűjtés
df = pd.read_csv("train.csv")

#%% Feltérképező elemzés - Kategorikus változók feltérképezése
print(df.describe())

distplot = sns.distplot(df["Burn Rate"],hist_kws={'edgecolor':'black', 'alpha':1.0}) 
plt.ylabel('Density')
plt.title('Kiégettség eloszlása')
plt.show()

fig = plt.figure(figsize=(12,8))

plt.subplot(1,3,1)
sns.countplot(x="Gender", hue="Company Type", data=df)

plt.subplot(1,3,2)
sns.countplot(x="Gender", hue="WFH Setup Available", data= df)

plt.subplot(1,3,3)
sns.countplot(x="Company Type", hue="WFH Setup Available", data= df)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.suptitle('Kategorikus jellemzők vizsgálata')
plt.show()


#%% Feltérképező elemzés - Numerikus változók feltérképezése
fig = plt.figure(figsize=(10,10))

plt.subplot(3,1,1)
sns.scatterplot(df["Burn Rate"],df["Designation"])  

plt.subplot(3,1,2)
sns.scatterplot(df["Burn Rate"],df["Resource Allocation"]) 

plt.subplot(3,1,3)
sns.scatterplot(df["Burn Rate"],df["Mental Fatigue Score"])
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.suptitle('Numerikus jellemzők vizsgálata')
plt.show()

#%% Korrelációs mátrix folytonos változókra
fig = plt.figure(figsize=(4,4))
mask = np.triu(np.ones_like(df[df.columns.values].corr(), dtype=bool))
sns.heatmap(df[df.columns.values].corr(),annot=True,mask=mask,fmt=".0%")
plt.title('Korrelációs mátrix')
plt.show() 

#%% Home office és kiégettség kapcsolata
fig = plt.figure(figsize=(8,6))
WFHplot = sns.countplot(x="Burn Rate", hue="WFH Setup Available", data= df)
WFHplot.axes.get_xaxis().set_ticks([])
plt.title('Home office és kiégettség kapcsolata') 

#%%Dátum átalakítás, ábrázolás
df["Date of Joining"] = pd.to_datetime(df["Date of Joining"]).dt.strftime("%Y%m%d")
datePlot = sns.scatterplot(x=np.array(df["Date of Joining"]),y=df["Burn Rate"],s=20)
datePlot.axes.get_xaxis().set_ticks([]) 
plt.xlabel("Date") 
plt.title('Csatlakozás dátuma és kiégettség kapcsolata')
plt.show()
 
#%% Adatelőkészítés - kategorikus változók kódolása
categorical_features = ["Gender", "Company Type", "WFH Setup Available"]
numerical_features = ["Designation", "Resource Allocation", "Mental Fatigue Score"]
print(df.isnull().sum())
print(df.duplicated().sum())  
# nincs se nan érték a kategorikus változóknál, se duplikáció tehát lehet őket kódolni
df = pd.get_dummies(df, columns=categorical_features,drop_first=True)

#%% Adatelőkészítés - numerikus változók átalakítása, nan értékek kezelése
print("Burn Rate hiányzó értékei",round(df["Burn Rate"].isnull().sum()/len(df)*100,2),"%")
print("Resource Allocation hiányzó értékei",round(df["Resource Allocation"].isnull().sum()/len(df)*100,2),"%")
print("Mental Fatigue Score hiányzó értékei",round(df["Mental Fatigue Score"].isnull().sum()/len(df)*100,2),"%")
df.dropna(subset=["Burn Rate","Resource Allocation","Mental Fatigue Score"], inplace=True)
print(df.isnull().sum())

df.drop(columns=["Employee ID","Date of Joining"], inplace=True)

#%% Lineáris regressziós modell - Egyváltozós
linmodel= LinearRegression()
#függő és független változó szétválasztása
X=df[["Mental Fatigue Score"]] 
y=df[["Burn Rate"]]

#tanuló és teszt adatok szétválasztása
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

linmodel.fit(X_train,y_train)

predictions = linmodel.predict(X_test)

print(r2_score(y_test,predictions)) #1 változóval 89%

plt.figure(figsize=(10,8))
plt.scatter(X_test, y_test,s=2)
plt.plot(X_test, predictions, color='#000', linewidth=2)
plt.xlabel('Mental Fatigue Score')
plt.ylabel('Burn Rate')
plt.title('Lineáris regressziós függvény')
plt.show()

#%% Lineáris regressziós modell - Többváltozós

X=df[numerical_features]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

linmodel.fit(X_train,y_train)

predictions = linmodel.predict(X_test)

print(r2_score(y_test,predictions)) #3 változóval 91%
#%% Logisztikus regressziós modell - előkészítés
#ha nagyobb a kiégettségi foka mint 0.5 akkor 1 azaz kiégettnek nyilvánítjuk, ha kisebb akkor 0 azaz nem kiégett
dfLR = df.copy()
dfLR["Burn Rate"] = [1 if x > 0.5 else 0 for x in dfLR["Burn Rate"]]
dfLR.rename(columns={"Burn Rate":"IsBurnedOut"},inplace=True)

ax = sns.countplot(x="IsBurnedOut", data=dfLR)
for p in ax.patches:
        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()+50))
plt.title('Kiégettek száma')

#mivel különböző mértékű skálákon vannak értékelve a különböző featureök(1-5;1-10) így indokolt a standardizálás
ss = StandardScaler() 
ss.fit(dfLR[numerical_features])
dfLR[numerical_features] = ss.transform(dfLR[numerical_features])
        
#%% Logisztikus regressziós modell tanítása, tesztelése
X=dfLR.drop("IsBurnedOut",axis=1)
y=dfLR["IsBurnedOut"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

model=LogisticRegression(solver='lbfgs', random_state=42, max_iter=1000)
model.fit(X_train,y_train)

predictions = model.predict(X_test)

conf_matrix = confusion_matrix(y_test,predictions)
plt.figure(figsize=(6,6))
ax = sns.heatmap(conf_matrix,annot=True,fmt="d")
plt.title('Konfúziós mátrix')
labels = ['Not Burned out','Burned out']
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
plt.xlabel('Jósolt')
plt.ylabel('Valódi')
plt.show()

print(round(accuracy_score(y_test,predictions)*100),"%")
#%% Jellemzők hatása a kiégettségre 
feature_importance = model.coef_[0]
plt.figure(figsize=(10,6))
plt.bar(X_train.columns.values, feature_importance)
plt.xticks(rotation='vertical')
plt.title('Egyes jellemzők hatása a kiégettségre')
plt.show()

#%% Logisztikus függvény vizualizálás a prezentációhoz (nem a valódi, mivel csak 1 változóval szemben ábrázoljuk, de szemléltetésre jó)
probabilities = model.predict_proba(X_train)
plt.figure(figsize=(10,8))
sns.regplot(x=X["Mental Fatigue Score"], y=model.predict_proba(X)[:,1],marker='o', data=dfLR, logistic= True,scatter_kws={'s':5},line_kws={'color':'black','linewidth':2});
plt.ylabel('IsBurnedOut prediction probabilities')
plt.title('Logisztikus regressziós függvény')
plt.show()

#%% Gradiens ereszkedés szemléltetés
# Paraméterek felvétele
x = df.loc[:, 'Mental Fatigue Score']
y = df.loc[:, 'Burn Rate']
b0 = 0
b1 = 0

#Egyenes egyenlete
f = b0 + b1*x

#Egyenes ábrázolása
def plotf():
    plt.figure(figsize=(12, 7))
    plt.scatter(x, y,s=4)
    plt.plot(x, f, color='red')
    plt.title('Egyváltozós lineáris regresszió')
    plt.xlabel('Mental Fatigue Score')
    plt.ylabel('Burn Rate')

plotf()



#%% b0 és b1 közelítése az optimálishoz
alpha = 0.0000001

b0 = b0 - np.sum(2*(b0 + b1*x - y))*alpha
b1 = b1 - np.sum(2*(b0 + b1*x - y)*x)*alpha

#Eltérés négyzet
print('b0: ', b0, ' b1: ', b1, 'SSR',np.sum((f-y)**2))


# Regresszió fgv újraszámolása
f = b0 + b1*x

plotf()

#%% Hiba számolása ciklikusan
b0=0
b1=0

hibakovetes = []
alpha = 0.0000001

for i in range(100):
   
    b0 = b0 - np.sum(2*(b0 + b1*x - y))*alpha
    b1 = b1 - np.sum(2*(b0 + b1*x - y)*x)*alpha
    hibakovetes.append(np.sum((b0 + b1*x - y)**2))
    f = b0 + b1*x
    plotf()
    
    
print('b0: ', b0, ' b1: ', b1)
print('hiba négyzetösszeg: ', np.sum((f - y)**2))

#%% Hiba ábrázolása
plt.figure(figsize=(12, 7))
plt.scatter(np.linspace(0, len(hibakovetes), len(hibakovetes)), hibakovetes)
plt.title('Négyzetes hiba epochonként')
plt.xlabel('Epoch')
plt.ylabel('Négyzetes hiba')
plt.show()


#%% Gradiens ereszkedések - Batch 
graddf = pd.DataFrame(data={"Mental Fatigue Score" : df["Mental Fatigue Score"], "Resource Allocation" : df["Resource Allocation"],"Burn Rate" : df["Burn Rate"]})


sx = StandardScaler() 
sy = StandardScaler() 

scaled_X = sx.fit_transform(graddf.drop("Burn Rate",axis="columns"))
scaled_y = sy.fit_transform(graddf["Burn Rate"].values.reshape(df.shape[0],1))

scaled_y.reshape(len(scaled_y),)

def batch_gradient_descent(X, y_true, epochs, learning_rate = 0.01):
    number_of_features = X.shape[1]
  
    w = np.ones(shape=(number_of_features)) 
    b = 0
    total_samples = X.shape[0] 
    
    cost_list = []
    epoch_list = []
    
    for i in range(epochs):        
        y_predicted = np.dot(w, X.T) + b

        w_grad = -(2/total_samples)*(X.T.dot(y_true-y_predicted))
        b_grad = -(2/total_samples)*np.sum(y_true-y_predicted)
        
        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad
        
        cost = np.mean(np.square(y_true-y_predicted)) 
        
        if i%10==0:
            cost_list.append(cost)
            epoch_list.append(i)
    
    return w, b, cost, cost_list, epoch_list

w, b, cost, cost_list, epoch_list = batch_gradient_descent(scaled_X,scaled_y.reshape(scaled_y.shape[0],),500)
w, b, cost

fig = plt.figure(figsize=(10,10))
ax1 = plt.subplot(3,1,1)
plt.plot(epoch_list,cost_list)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.suptitle('Gradiens ereszkedések')
ax1.title.set_text('Batch')
plt.ylabel("cost")


# Gradiens ereszkedések - Sztochasztikus

import random
random.randint(0,6)

def stochastic_gradient_descent(X, y_true, epochs, learning_rate = 1):
    number_of_features = X.shape[1]
    w = np.ones(shape=(number_of_features)) 
    b = 0
    total_samples = X.shape[0]
    
    cost_list = []
    epoch_list = []
    
    for i in range(epochs):    
        random_index = random.randint(0,total_samples-1) 
        sample_x = X[random_index]
        sample_y = y_true[random_index]
        
        y_predicted = np.dot(w, sample_x.T) + b
    
        w_grad = -(2/total_samples)*(sample_x.T.dot(sample_y-y_predicted))
        b_grad = -(2/total_samples)*(sample_y-y_predicted)
        
        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad
        
        cost = np.square(sample_y-y_predicted)
        
        if i%100==0: 
            cost_list.append(cost)
            epoch_list.append(i)
        
    return w, b, cost, cost_list, epoch_list

w_sgd, b_sgd, cost_sgd, cost_list_sgd, epoch_list_sgd = stochastic_gradient_descent(scaled_X,scaled_y.reshape(scaled_y.shape[0],),10000)
w_sgd, b_sgd, cost_sgd


ax2 = plt.subplot(3,1,2)
plt.plot(epoch_list_sgd,cost_list_sgd)
plt.ylabel("cost")
ax2.set_title('Stochastic')


# Gradiens ereszkedések - Mini-batch
def mini_batch_gradient_descent(X, y_true, epochs = 100, batch_size = 5, learning_rate = 0.01):
    number_of_features = X.shape[1]
   
    w = np.ones(shape=(number_of_features)) 
    b = 0
    total_samples = X.shape[0] 
    
    if batch_size > total_samples: 
        batch_size = total_samples
        
    cost_list = []
    epoch_list = []
    
    num_batches = int(total_samples/batch_size)
    
    for i in range(epochs):    
        random_indices = np.random.permutation(total_samples)
        X_tmp = X[random_indices]
        y_tmp = y_true[random_indices]
        
        for j in range(0,total_samples,batch_size):
            Xj = X_tmp[j:j+batch_size]
            yj = y_tmp[j:j+batch_size]
            y_predicted = np.dot(w, Xj.T) + b
            
            w_grad = -(2/len(Xj))*(Xj.T.dot(yj-y_predicted))
            b_grad = -(2/len(Xj))*np.sum(yj-y_predicted)
            
            w = w - learning_rate * w_grad
            b = b - learning_rate * b_grad
                
            cost = np.mean(np.square(yj-y_predicted))
        
        if i%10==0:
            cost_list.append(cost)
            epoch_list.append(i)
        
    return w, b, cost, cost_list, epoch_list

w, b, cost, cost_list, epoch_list = mini_batch_gradient_descent(
    scaled_X,
    scaled_y.reshape(scaled_y.shape[0],),
    epochs = 120,
    batch_size = 5
)
w, b, cost


ax3 = plt.subplot(3,1,3)
plt.plot(epoch_list,cost_list)
ax3.title.set_text('Mini batch')
plt.xlabel("epoch")
plt.ylabel("cost")
plt.show()

