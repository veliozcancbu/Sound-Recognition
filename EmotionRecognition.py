import pandas as pd
import numpy as np
import librosa
import os
from glob import glob
import wave
import matplotlib.pyplot as plt
import IPython.display as ipd
import seaborn as sns
from plotly import express as px
import plotly
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score

Ravdess='/kaggle/input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/'

ravdess_directory_list=os.listdir(Ravdess)


file_emotion=[]
file_path=[]

for dir in ravdess_directory_list:
    actor=os.listdir(Ravdess + dir)
    for file in actor:
        part=file.split('.')[0]
        part=part.split('-')
        file_emotion.append(int(part[2]))
        file_path.append(Ravdess + dir + '/' + file)

emotion_df=pd.DataFrame(file_emotion, columns=['Emotions'])

path_df=pd.DataFrame(file_path, columns=['Path'])
Ravdess_df=pd.concat([emotion_df,path_df],axis=1)

Ravdess_df.Emotions.replace({1:'Doğal', 2:'Sakin', 3:'Mutlu', 4:'Üzgün', 5:'Kızgın', 6:'Korkmuş', 7:'İğrenmiş', 8:'Şaşırmış'}, inplace=True)
Ravdess_df.head()

fig=px.histogram(Ravdess_df,x='Emotions',title='Duyguların toplam sayısı')
fig.update_layout(xaxis_title='Emotions',yaxis_title='Count',showlegend=False)
#fig.update_traces(marker_color='steelblue')
fig.show()

plotly.offline.plot(fig, filename='Duygutablosu.html')

emotion='Doğal'
path=np.array(Ravdess_df.Path[Ravdess_df.Emotions==emotion])[1]

y, sr= librosa.load(path)
print(f'First 20 values of y:{y[:20]}') #Means almost first 20 ms
print(f'Shape of y: {y.shape}')
print(f'Sample rate: {sr}')
duration=librosa.get_duration(y=y,sr=sr)
print(f'The file duration is {duration} seconds')
pd.Series(y).plot(figsize=(15,5),lw=1,title='Doğal saf ses örneği')
plt.show()
pd.Series(y[40000:40100]).plot(figsize=(15,5), lw=1, title='Doğal ses örneği(Yakınlaştırılmış)')
plt.show()
#Spectogram and a mathematical parts.
d=librosa.stft(y) #Fourier transform here
S_db=librosa.amplitude_to_db(np.abs(d), ref=np.max) #Changing ampitude to decibel.
a,ax=plt.subplots(figsize=(15,5))
img=librosa.display.specshow(S_db,
                            x_axis='time',
                            y_axis='log',#wave form after transformation.
                            ax=ax)
ax.set_title('Spectogram Örneği', fontsize=20)
a.colorbar(img, ax=ax,format=f'%0.2f')
plt.show()
ipd.Audio(path)
#Sondaki  grafikteki her bir farklılık bir öznitelik taşır. Bunu resimlerdei pixel gibi düşünebiliriz.


emotion='Şaşırmış'
path=np.array(Ravdess_df.Path[Ravdess_df.Emotions==emotion])[1]

y, sr= librosa.load(path)
print(f'First 20 values of y:{y[:20]}') #Means almost first 20 ms
print(f'Shape of y: {y.shape}')
print(f'Sample rate: {sr}')
duration=librosa.get_duration(y=y,sr=sr)
print(f'The file duration is {duration} seconds')
pd.Series(y).plot(figsize=(15,5),lw=1,title='Şaşırmış saf ses örneği')
plt.show()
pd.Series(y[40000:40100]).plot(figsize=(15,5), lw=1, title='Şaşırmış ses örneği(Yakınlaştırılmış)')
plt.show()
#Spectogram and a mathematical parts.
d=librosa.stft(y) #Fourier transform here
S_db=librosa.amplitude_to_db(np.abs(d), ref=np.max) #Changing ampitude to decibel.
a,ax=plt.subplots(figsize=(15,5))
img=librosa.display.specshow(S_db,
                            x_axis='time',
                            y_axis='log',#wave form after transformation.
                            ax=ax)
ax.set_title('Spectogram Örneği', fontsize=20)
a.colorbar(img, ax=ax,format=f'%0.2f')
plt.show()
ipd.Audio(path)
#Sondaki  grafikteki her bir farklılık bir öznitelik taşır. Bunu resimlerdei pixel gibi düşünebiliriz.


emotion='İğrenmiş'
path=np.array(Ravdess_df.Path[Ravdess_df.Emotions==emotion])[1]

y, sr= librosa.load(path)
print(f'First 20 values of y:{y[:20]}') #Means almost first 20 ms
print(f'Shape of y: {y.shape}')
print(f'Sample rate: {sr}')
duration=librosa.get_duration(y=y,sr=sr)
print(f'The file duration is {duration} seconds')
pd.Series(y).plot(figsize=(15,5),lw=1,title='İğrenmiş saf ses örneği')
plt.show()
pd.Series(y[40000:40100]).plot(figsize=(15,5), lw=1, title='İğrenmiş ses örneği(Yakınlaştırılmış)')
plt.show()
#Spectogram and a mathematical parts.
d=librosa.stft(y) #Fourier transform here
S_db=librosa.amplitude_to_db(np.abs(d), ref=np.max) #Changing ampitude to decibel.
a,ax=plt.subplots(figsize=(15,5))
img=librosa.display.specshow(S_db,
                            x_axis='time',
                            y_axis='log',#wave form after transformation.
                            ax=ax)
ax.set_title('Spectogram Örneği', fontsize=20)
a.colorbar(img, ax=ax,format=f'%0.2f')
plt.show()
ipd.Audio(path)
#Sondaki  grafikteki her bir farklılık bir öznitelik taşır. Bunu resimlerdei pixel gibi düşünebiliriz.


emotion='Korkmuş'
path=np.array(Ravdess_df.Path[Ravdess_df.Emotions==emotion])[1]

y, sr= librosa.load(path)
print(f'First 20 values of y:{y[:20]}') #Means almost first 20 ms
print(f'Shape of y: {y.shape}')
print(f'Sample rate: {sr}')
duration=librosa.get_duration(y=y,sr=sr)
print(f'The file duration is {duration} seconds')
pd.Series(y).plot(figsize=(15,5),lw=1,title='Korkmuş saf ses örneği')
plt.show()
pd.Series(y[40000:40100]).plot(figsize=(15,5), lw=1, title='Korkmuş ses örneği(Yakınlaştırılmış)')
plt.show()
#Spectogram and a mathematical parts.
d=librosa.stft(y) #Fourier transform here
S_db=librosa.amplitude_to_db(np.abs(d), ref=np.max) #Changing ampitude to decibel.
a,ax=plt.subplots(figsize=(15,5))
img=librosa.display.specshow(S_db,
                            x_axis='time',
                            y_axis='log',#wave form after transformation.
                            ax=ax)
ax.set_title('Spectogram Örneği', fontsize=20)
a.colorbar(img, ax=ax,format=f'%0.2f')
plt.show()
ipd.Audio(path)
#Sondaki  grafikteki her bir farklılık bir öznitelik taşır. Bunu resimlerdei pixel gibi düşünebiliriz.


emotion='Üzgün'
path=np.array(Ravdess_df.Path[Ravdess_df.Emotions==emotion])[1]

y, sr= librosa.load(path)
print(f'First 20 values of y:{y[:20]}') #Means almost first 20 ms
print(f'Shape of y: {y.shape}')
print(f'Sample rate: {sr}')
duration=librosa.get_duration(y=y,sr=sr)
print(f'The file duration is {duration} seconds')
pd.Series(y).plot(figsize=(15,5),lw=1,title='Üzgün saf ses örneği')
plt.show()
pd.Series(y[40000:40100]).plot(figsize=(15,5), lw=1, title='Üzgün ses örneği(Yakınlaştırılmış)')
plt.show()
#Spectogram and a mathematical parts.
d=librosa.stft(y) #Fourier transform here
S_db=librosa.amplitude_to_db(np.abs(d), ref=np.max) #Changing ampitude to decibel.
a,ax=plt.subplots(figsize=(15,5))
img=librosa.display.specshow(S_db,
                            x_axis='time',
                            y_axis='log',#wave form after transformation.
                            ax=ax)
ax.set_title('Spectogram Örneği', fontsize=20)
a.colorbar(img, ax=ax,format=f'%0.2f')
plt.show()
ipd.Audio(path)
#Sondaki  grafikteki her bir farklılık bir öznitelik taşır. Bunu resimlerdei pixel gibi düşünebiliriz.


emotion='Sakin'
path=np.array(Ravdess_df.Path[Ravdess_df.Emotions==emotion])[1]

y, sr= librosa.load(path)
print(f'First 20 values of y:{y[:20]}') #Means almost first 20 ms
print(f'Shape of y: {y.shape}')
print(f'Sample rate: {sr}')
duration=librosa.get_duration(y=y,sr=sr)
print(f'The file duration is {duration} seconds')
pd.Series(y).plot(figsize=(15,5),lw=1,title='Sakin saf ses örneği')
plt.show()
pd.Series(y[40000:40100]).plot(figsize=(15,5), lw=1, title='Sakin ses örneği(Yakınlaştırılmış)')
plt.show()
#Spectogram and a mathematical parts.
d=librosa.stft(y) #Fourier transform here
S_db=librosa.amplitude_to_db(np.abs(d), ref=np.max) #Changing ampitude to decibel.
a,ax=plt.subplots(figsize=(15,5))
img=librosa.display.specshow(S_db,
                            x_axis='time',
                            y_axis='log',#wave form after transformation.
                            ax=ax)
ax.set_title('Spectogram Örneği', fontsize=20)
a.colorbar(img, ax=ax,format=f'%0.2f')
plt.show()
ipd.Audio(path)
#Sondaki  grafikteki her bir farklılık bir öznitelik taşır. Bunu resimlerdei pixel gibi düşünebiliriz.


emotion='Mutlu'
path=np.array(Ravdess_df.Path[Ravdess_df.Emotions==emotion])[1]

y, sr= librosa.load(path)
print(f'First 20 values of y:{y[:20]}') #Means almost first 20 ms
print(f'Shape of y: {y.shape}')
print(f'Sample rate: {sr}')
duration=librosa.get_duration(y=y,sr=sr)
print(f'The file duration is {duration} seconds')
pd.Series(y).plot(figsize=(15,5),lw=1,title='Mutlu saf ses örneği')
plt.show()
pd.Series(y[40000:40100]).plot(figsize=(15,5), lw=1, title='Mutlu ses örneği(Yakınlaştırılmış)')
plt.show()
#Spectogram and a mathematical parts.
d=librosa.stft(y) #Fourier transform here
S_db=librosa.amplitude_to_db(np.abs(d), ref=np.max) #Changing ampitude to decibel.
a,ax=plt.subplots(figsize=(15,5))
img=librosa.display.specshow(S_db,
                            x_axis='time',
                            y_axis='log',#wave form after transformation.
                            ax=ax)
ax.set_title('Spectogram Örneği', fontsize=20)
a.colorbar(img, ax=ax,format=f'%0.2f')
plt.show()
ipd.Audio(path)
#Sondaki  grafikteki her bir farklılık bir öznitelik taşır. Bunu resimlerdei pixel gibi düşünebiliriz.


emotion='Kızgın'
path=np.array(Ravdess_df.Path[Ravdess_df.Emotions==emotion])[1]

y, sr= librosa.load(path)
print(f'First 20 values of y:{y[:20]}') #Means almost first 20 ms
print(f'Shape of y: {y.shape}')
print(f'Sample rate: {sr}')
duration=librosa.get_duration(y=y,sr=sr)
print(f'The file duration is {duration} seconds')
pd.Series(y).plot(figsize=(15,5),lw=1,title='Kızgın saf ses örneği')
plt.show()
pd.Series(y[40000:40100]).plot(figsize=(15,5), lw=1, title='Kızgın ses örneği(Yakınlaştırılmış)')
plt.show()
#Spectogram and a mathematical parts.
d=librosa.stft(y) #Fourier transform here
S_db=librosa.amplitude_to_db(np.abs(d), ref=np.max) #Changing ampitude to decibel.
a,ax=plt.subplots(figsize=(15,5))
img=librosa.display.specshow(S_db,
                            x_axis='time',
                            y_axis='log',#wave form after transformation.
                            ax=ax)
ax.set_title('Spectogram Örneği', fontsize=20)
a.colorbar(img, ax=ax,format=f'%0.2f')
plt.show()
ipd.Audio(path)
#Sondaki  grafikteki her bir farklılık bir öznitelik taşır. Bunu resimlerdei pixel gibi düşünebiliriz.


def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data
def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)
def shift(data):
    shift_range=int(np.random.uniform(low=5,high=5)*1000)
    return np.roll(data, shift_range)
def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

pd.Series(y).plot(figsize=(15,5),lw=1, title='Temiz ses')
ipd.Audio(path)

x=stretch(y)
pd.Series(x).plot(figsize=(15,5),lw=1, title='Stretched sound')
ipd.Audio(x, rate=sr)

x=shift(y)
pd.Series(x).plot(figsize=(15,5),lw=1, title='Shifted Sound')
ipd.Audio(x,rate=sr)

x=pitch(y,sr)
pd.Series(x).plot(figsize=(15,5),lw=1,title='Pitched Sound')
ipd.Audio(x,rate=sr)

def feature_extractors(data):
    result=np.array([])
    zcr=np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result,zcr))

    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    result = np.hstack((result, chroma_stft))

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr).T, axis=0)
    result = np.hstack((result, mfcc))

    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
    result = np.hstack((result, mel))

    return result

def get_features (path):
    y, sr =librosa.load(path, duration=2.5, offset=0.6)
    res1=feature_extractors(y)
    result=np.array(res1)

    noise_data=noise(y)
    res2=feature_extractors(noise_data)
    result=np.vstack((result,res2))

    new_data=stretch(y)
    data_stretch_pitch=pitch(new_data, sr)
    res3=feature_extractors(data_stretch_pitch)
    result=np.vstack((result,res3))


    return result


X, Y = [],[]
for path, emotion in tqdm(zip(Ravdess_df.Path,Ravdess_df.Emotions)):
    feature=get_features(path)
    for ele in feature:
        X.append(ele)
        Y.append(emotion)

Features=pd.DataFrame(X)
Features['Labels']=Y
Features.to_csv('features.csv',index=False)
Features.tail()

X=Features.iloc[ : , : -1].values
Y=Features['Labels'].values


encoder= OneHotEncoder()
Y=encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

x_train, x_test, y_train, y_test=train_test_split(X, Y, random_state=0, shuffle=True)
x_train.shape, y_train.shape, y_test.shape, x_test.shape

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)
x_train.shape, y_train.shape, x_test.shape,  y_test.shape

y_test.shape

x_train.shape[1]

num_labels = 8

model=Sequential()

model.add(Dense(125, input_shape=(x_train.shape[1],)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(250))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(125))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

epoch=500
num_batch_size=32

model1=model.fit(x_train, y_train, batch_size=num_batch_size, epochs=epoch, validation_data=(x_test,y_test), verbose=1)

val_test_acc=model.evaluate(x_test,y_test, verbose=0)
print(val_test_acc[1])

plt.figure(figsize=(7,7))
plt.grid()
plt.plot(model1.history['loss'])
plt.plot(model1.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Training','Validation'], loc='upper right')
plt.savefig("loss_curve.pdf")
plt.show()


plt.figure(figsize=(5,5))
plt.ylim(0,1.1)
plt.grid()
plt.plot(model1.history['accuracy'])
plt.plot(model1.history['val_accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Training','Validation'])
plt.savefig("acc_curve.pdf")
plt.show()

result_array=model.predict(x_test)

result_classes = ['Doğal','Sakin','Mutlu','Üzgün','Kızgın','Korkmuş','İğrenmiş','Şaşırmış']

result=np.argmax(result_array[13])
print(result_classes[result])

Y_pred_test=model.predict(x_test)
Y_pred_test=np.argmax(Y_pred_test,axis=1)
classification= classification_report(np.argmax(y_test,axis=1),Y_pred_test)
print(classification)

pred_test = model.predict(x_test)
y_pred = encoder.inverse_transform(pred_test)

y_test = encoder.inverse_transform(y_test)

df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
df['Predicted Labels'] = y_pred.flatten()
df['Actual Labels'] = y_test.flatten()

df.head(20)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize = (12, 5))
cm = pd.DataFrame(cm , index = [i for i in encoder.categories_] , columns = [i for i in encoder.categories_])
sns.heatmap(cm, linecolor='black', cmap='YlOrRd', linewidth=1, annot=True, fmt='')
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=14)
plt.ylabel('Actual Labels', size=14)
plt.show()

model.save('emotionmodel.hdf5')

