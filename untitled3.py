from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

def def_model():
    model=Sequential()
    model.add(Conv2D(32,(3,3),activation='relu',kernel_initializer='he_uniform',input_shape=(28,28,1)))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64,(3,3),activation='relu',kernel_initializer='he_uniform'))
    model.add(Conv2D(64,(3,3),activation='relu',kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(100,activation='relu',kernel_initializer='he_uniform'))
    model.add(Dense(10,activation='softmax'))
    opt=SGD(lr=0.01,momentum=0.9)
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
    return model
def evaluate_model(model,trainX,Y,n_folds=5):
    scores=[]
    hist=[]
    kfold=KFold(n_folds,shuffle=True,random_state=1)
    for train_ix, test_ix in kfold.split(trainX):
        train_X,train_y,test_X,test_y=trainX[train_ix],Y[train_ix],trainX[test_ix],Y[test_ix]
        hi=model.fit(train_X,train_y,epochs=10,batch_size=32,validation_data=(test_X,test_y),verbose=0)
        acc=model.evaluate(test_X,test_y,verbose=0)
        print(acc)
        scores.append(acc)
        hist.append(hi)
    return scores, hist, sum(scores)/len(scores)
    
    
from keras.datasets import mnist
from matplotlib import pyplot

(trainX,trainy), (testX,testy)=mnist.load_data()
(train__X,train__y), (test__X,test__y)=mnist.load_data()
for i in range(9):
    pyplot.subplot(440+i+1)
    pyplot.imshow(trainX[i],cmap=pyplot.get_cmap('gray'))
pyplot.show()

from keras.utils import to_categorical


trainX=trainX.reshape(trainX.shape[0],28,28,1)
testX=testX.reshape(testX.shape[0],28,28,1)
trainy=to_categorical(trainy)
testy=to_categorical(testy)


def prep(trainX,testX):
    trainX_norm=trainX.astype('float32')
    testX_norm=testX.astype('float32')
    trainX_norm=trainX_norm/255
    testX_norm=testX_norm/255
    return trainX_norm,testX_norm
trainX,testX=prep(trainX,testX)
model=def_model()

model.fit(trainX,trainy,epochs=10,batch_size=32,verbose=0)
model.save('final_m.h5')
    
    
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
def load_image(filename):
    img=load_img(filename,grayscale=True,target_size=(28,28))
    pyplot.subplot(330+0+1)
    pyplot.imshow(img,cmap=pyplot.get_cmap('gray'))
    pyplot.show()
    img=img_to_array(img)
    img=img.reshape(1,28,28,1)
    img=img.astype('float32')
    img=img/255
    return img

from keras.models import load_model
def pred(img,model=load_model('final_m.h5')):
    digit=model.predict_classes(img)
    return digit


for i in range(4):
    img=load_image('test/s_d'+str(i)+'.png')
    print(pred(img))

img=load_image('ample_data.png')
print(pred(img))





    
    