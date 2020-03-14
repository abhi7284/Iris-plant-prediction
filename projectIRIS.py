from tkinter import *
import tkinter.messagebox as m
w=Tk()
w.configure(bg="seagreen")
w.title("IRIS FLOWER PREDICTION")






#Algorithm

#step1-load data
#step2-divide input(X) and target(Y) 
#step3-split the data by using --> X_train,X_test,Y_train,Y_teat=train_test_split(X,Y,test_size=0.2)#random_state='3'

#step4-import module and create model
#step5-train the model .fit(X_train,Y_train)
#step6-test_pred=model.predict(X_test)
#step7-check accuracy score -> from sklearn.metrics import accuracy_score -> acc_knn=accuracy_score(Y_test,test_pred)

###########################################################################################

####################### Calculating ###########################################################################


#importing dataset into the program


from sklearn.datasets import load_iris

iris=load_iris()


#divide the datasets into X(input) and Y(output)

X=iris.data
Y=iris.target


#Split the data into 2 pairs
#-Training
#-Testing



from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=7)



#create a model

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

K=KNeighborsClassifier(n_neighbors=7)
LR=LogisticRegression(solver='liblinear',multi_class="auto")
DT=DecisionTreeClassifier()
NB=GaussianNB()



# Train the model

K.fit(X_train,Y_train) ##K nearest Neighbour

LR.fit(X_train,Y_train)## Logistic Regression

DT.fit(X_train,Y_train) ##Decision Tree

NB.fit(X_train,Y_train) ## Naive Bayes


#Test the model

Y_pred_knn=K.predict(X_test)

logPredict=LR.predict(X_test)

DTPredict=DT.predict(X_test)

NBPredict=NB.predict(X_test)



### Find the accuracy Model

from sklearn.metrics import accuracy_score

acc_knn=accuracy_score(Y_test,Y_pred_knn)
acc_knn=round(acc_knn*100,2)

acc_log=accuracy_score(Y_test,logPredict)
acc_log=round(acc_log*100,2)

acc_dt=accuracy_score(Y_test,DTPredict)
acc_dt=round(acc_dt*100,2)

acc_nb=accuracy_score(Y_test,NBPredict)
acc_nb=round(acc_nb*100,2)




################################  * END *  #######################################


################################  * Methods * ####################################


def cal_knn():
    global acc_knn
    nn=str(acc_knn)+'%'
    m.showinfo(title='KNN',message=nn)
    

def cal_logR():
    global acc_log
    lg=str(acc_log)+'%'
    m.showinfo(title='LR',message=lg)

def cal_dt():
    global acc_dt
    dt=str(acc_dt)+'%'
    m.showinfo(title='DT',message=dt)
    

def cal_nb():
    global acc_nb
    nb=str(acc_nb)+'%'
    m.showinfo(title='NB',message=nb)

####################

    
def reset():
    '''Reset all the entry '''
    
    v6.set("")
    v7.set("")
    v8.set("")
    v9.set("")
    v10.set("")



def max_accuracy():
    
    '''returns the max accurate model'''
    
    global acc_knn,acc_log,acc_dt,acc_nb
    global K,LR,DT,NV

    a=max(acc_knn,acc_log,acc_dt,acc_nb)
    if a==acc_knn:
        return K
    
    elif a==acc_log:
        return LR
    
    elif a==acc_dt:
        return DT
    
    else:
        return NB
        
    

def predict():
    '''predict the iris flower type from the entered data'''
    
    global K
    
    n1=float(v6.get())
    n2=float(v7.get())
    n3=float(v8.get())
    n4=float(v9.get())
    
    L=[n1,n2,n3,n4]
    print(L)

    ma=max_accuracy()
    type(ma)
    
    pred=ma.predict([L])

    #virginica
    #setosa
    #versicolor

    if pred[0]==0:
        v10.set("IRIS-SETOSA")
    elif pred[0]==1:
        v10.set("IRIS-VERSICOLOR")
    else:
        v10.set("IRIS-VIRGINICA")

    



def compare():
    
    import matplotlib.pyplot as plt

    model=['K','LR','DT','NB']
    accuracy=[acc_knn,acc_log,acc_dt,acc_nb]
    

    plt.bar(model,accuracy,color=["red","blue","yellow","green"])
    plt.title('Model Comparison')
    plt.xlabel("MODELS")
    plt.ylabel("ACCURACY")
    plt.show()
    
    





    

############################################################################


############################### * GUI * ##################################

#Buttons

 #column 1
Bknn=Button(w,text="KNN",bg="slateblue",font=("arial",10,"bold"),width=4,height=1,bd=8,command=cal_knn)
Blg=Button(w,text="LG",bg="slateblue",font=("arial",10,"bold"),width=4,height=1,bd=8,command=cal_logR)
Bdt=Button(w,text="DT",bg="slateblue",font=("arial",10,"bold"),width=4,height=1,bd=8,command=cal_dt)
Bnb=Button(w,text="NB",bg="slateblue",font=("arial",10,"bold"),width=4,height=1,bd=8,command=cal_nb)
Bcompare=Button(w,text="Compare",bg="slateblue",font=("arial",11,"bold"),bd=8,command=compare)

#row 6
Bpredict=Button(w,text="Predict",bg="slateblue",font=("arial",11,"bold"),bd=8,command=predict)
Breset=Button(w,text="Reset",bg="slateblue",font=("arial",8,"bold"),bd=8,command=reset)


#Labels
img=PhotoImage(file="D:/mydatabase/iris_flower2.png")

L=Label(w,image=img)
L.grid(row=0,column=0,columnspan=6,padx=(2,2),pady=(5,5))
lhead=Label(w,text="IRIS FLOWER PREDICTION",bg="seagreen",fg="navy",font=("arial",15,"bold"))
sl=Label(w,text="Sepal Length",bg="seagreen",font=("arial",10,"bold"))
pl=Label(w,text="Sepal Width",bg="seagreen",font=("arial",10,"bold"))
sw=Label(w,text="Petal Length",bg="seagreen",font=("arial",10,"bold"))
pw=Label(w,text="Petal Width",bg="seagreen",font=("arial",10,"bold"))



#Entry




#v = StringVar(w, value='0')


v6=StringVar(w, value='0')
v7=StringVar(w, value='0')
v8=StringVar(w, value='0')
v9=StringVar(w, value='0')
v10=StringVar(w, value='0')


e1=Entry(w,bg="grey65",font=("arial",10,"bold"),fg='black',justify="right",bd=8,textvariable=v6)
e2=Entry(w,bg="grey65",font=("arial",10,"bold"),fg='black',justify="right",bd=8,textvariable=v7)
e3=Entry(w,bg="grey65",font=("arial",10,"bold"),fg='black',justify="right",bd=8,textvariable=v8)
e4=Entry(w,bg="grey65",font=("arial",10,"bold"),fg='black',justify="right",bd=8,textvariable=v9)

epd=Entry(w,bg="grey60",font=("arial",10,"bold"),fg='black',justify="right",bd=8,textvariable=v10)






# Arrangements

lhead.grid(row=1,column=1,columnspan=5,padx=(10,10),pady=(10,10))
Bknn.grid(row=2,column=1,padx=(10,10),pady=(10,10))
Blg.grid(row=3,column=1,padx=(10,10),pady=(10,10))
Bdt.grid(row=4,column=1,padx=(10,10),pady=(10,10))
Bnb.grid(row=5,column=1,padx=(10,10),pady=(10,10))
Bcompare.grid(row=6,column=1,padx=(10,10),pady=(10,10),columnspan=1)

Bpredict.grid(row=6,column=3,padx=(10,10),pady=(10,10))
epd.grid(row=6,column=4,padx=(10,10),pady=(10,10))
Breset.grid(row=6,column=5,padx=(0,10),pady=(10,10))

sl.grid(row=2,column=3,padx=(10,10),pady=(10,10))
sw.grid(row=3,column=3,padx=(10,10),pady=(10,10))
pl.grid(row=4,column=3,padx=(10,10),pady=(10,10))
pw.grid(row=5,column=3,padx=(10,10),pady=(10,10))

e1.grid(row=2,column=4,padx=(10,10),pady=(10,10))
e2.grid(row=3,column=4,padx=(10,10),pady=(10,10))
e3.grid(row=4,column=4,padx=(10,10),pady=(10,10))
e4.grid(row=5,column=4,padx=(10,10),pady=(10,10))




w.mainloop()

#######################################################################################
