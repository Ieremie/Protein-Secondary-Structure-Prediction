LR = 0.0005
drop_out = 0.3
batch_dim = 64
nn_epochs = 1
w_reg = regularizers.l2(0.0001)
number_filters = 16

loss = 'categorical_crossentropy'


m = Sequential()

#first convolutional neural netwok
m.add(Conv1D( 16 , 3,  strides=1, padding='same', activation='relu', use_bias=True, input_shape=(windowSize, 21), kernel_regularizer=w_reg))
m.add(BatchNormalization())

m.add(Conv1D( 16, 3,  strides=1, padding='same', activation='relu', use_bias=True, kernel_regularizer=w_reg))
m.add(BatchNormalization())
m.add(Conv1D( 16, 5,  strides=1, padding='same', activation='relu', use_bias=True, kernel_regularizer=w_reg))
m.add(BatchNormalization())
m.add(Conv1D( 16, 7,  strides=1, padding='same', activation='relu', use_bias=True, kernel_regularizer=w_reg))
m.add(BatchNormalization())


m.add(Conv1D( 16, 3,  strides=1, padding='same', activation='relu', use_bias=True, kernel_regularizer=w_reg))
m.add(BatchNormalization())
m.add(Conv1D( 16, 5,  strides=1, padding='same', activation='relu', use_bias=True, kernel_regularizer=w_reg))
m.add(BatchNormalization())
m.add(Conv1D( 16, 7,  strides=1, padding='same', activation='relu', use_bias=True, kernel_regularizer=w_reg))
m.add(BatchNormalization())

m.add(Conv1D( 16, 3,  strides=1, padding='same', activation='relu', use_bias=True, kernel_regularizer=w_reg))
m.add(BatchNormalization())
m.add(Conv1D( 16, 5,  strides=1, padding='same', activation='relu', use_bias=True, kernel_regularizer=w_reg))
m.add(BatchNormalization())
m.add(Conv1D( 16, 7,  strides=1, padding='same', activation='relu', use_bias=True, kernel_regularizer=w_reg))
m.add(BatchNormalization())



m.add(Flatten())

#4 dense layer
m.add(Dense(200, activation='relu', use_bias=True,  kernel_regularizer=w_reg))


#5 softmax output layer
m.add(Dense(num_classes, activation = 'softmax'))

opt = optimizers.Adam(lr=LR)
m.compile(optimizer=opt, loss=loss,metrics=['accuracy', 'mae'])

print("\nHyper Parameters\n")
print("Learning Rate: " + str(LR))
print("Drop out: " + str(drop_out))
print("Batch dim: " + str(batch_dim))
print("Number of epochs: " + str(nn_epochs))
print("Regularizers: " + str(w_reg.l2))
print("\nLoss: " + loss + "\n")
m.summary()


#####################################


LR = 0.0005
drop_out = 0.3
batch_dim = 64
nn_epochs = 10
w_reg = regularizers.l2(0.0001)
number_filters = 16

loss = 'categorical_crossentropy'


input_shape = (windowSize, 21)

conv1_input = Input(shape=(windowSize, 21), name='InputWindow')

conv_1 = Conv1D( 64, 3,  strides=1, padding='same', activation='relu', use_bias=True, kernel_regularizer=w_reg)(conv1_input)
conv_1 = BatchNormalization(name='BN1')(conv_1)
conv_2 = Conv1D( 64, 3, strides=1, padding='same', activation='relu', use_bias=True,kernel_regularizer=w_reg)(conv_1)
conv_2 = BatchNormalization(name='BN2')(conv_2)
conv_3 = Conv1D( 64, 3,  strides=1, padding='same', activation='relu', use_bias=True,kernel_regularizer=w_reg)(conv_2)
conv_3 = BatchNormalization(name='BN3')(conv_3)



conv_4 = Conv1D( 64, 3, strides=1, padding='same', activation='relu', use_bias=True, kernel_regularizer=w_reg)(conv_3)
conv_4 = BatchNormalization(name='BN4')(conv_4)
conv_5 = Conv1D( 64, 3,  strides=1, padding='same', activation='relu', use_bias=True,kernel_regularizer=w_reg)(conv_4)
conv_5 = BatchNormalization(name='BN5')(conv_5)
conv_6 = Conv1D( 64, 3, strides=1, padding='same', activation='relu', use_bias=True,kernel_regularizer=w_reg)(conv_5)
conv_6 = BatchNormalization(name='BN6')(conv_6)


conv_7 = Conv1D( 64, 3,  strides=1, padding='same', activation='relu', use_bias=True, kernel_regularizer=w_reg)(conv_6)
conv_7 = BatchNormalization(name='BN7')(conv_7)
conv_8 = Conv1D( 64, 3,  strides=1, padding='same', activation='relu', use_bias=True,kernel_regularizer=w_reg)(conv_7)
conv_8 = BatchNormalization(name='BN8')(conv_8)
conv_9 = Conv1D( 64, 3,  strides=1, padding='same', activation='relu', use_bias=True,kernel_regularizer=w_reg)(conv_8)
conv_9 = BatchNormalization(name='BN9')(conv_9)


flatten  = Flatten()(conv_9)
first_dense = Dense(16, activation='relu', use_bias=True,  kernel_regularizer=w_reg, name='last')(flatten)
first_dense = BatchNormalization(name='BN10')(first_dense)
final_model_output = Dense(num_classes, activation = 'softmax', name='softmax')(first_dense)

m = Model(inputs=conv1_input, outputs=final_model_output)

opt = Adam(lr=LR)
m.compile(optimizer=opt, loss=loss,metrics=['accuracy', 'mae'])

print("\nHyper Parameters\n")
print("Learning Rate: " + str(LR))
print("Drop out: " + str(drop_out))
print("Batch dim: " + str(batch_dim))
print("Number of epochs: " + str(nn_epochs))
print("Regularizers: " + str(w_reg.l2))
print("\nLoss: " + loss + "\n")
m.summary()

#import os
#os.environ["PATH"] += os.pathsep + 'C:/Users/Ieremie/Anaconda3/pkgs/graphviz-2.38-hfd603c8_2/Library/bin'

plot_model(m)#, to_file='model.png')








####################################












LR = 0.0005
drop_out = 0.3
batch_dim = 64
nn_epochs = 10
w_reg = regularizers.l2(0.0001)
number_filters = 16

loss = 'categorical_crossentropy'



input_shape = (windowSize, 21)

conv1_input = Input(shape=(windowSize, 21), name='InputWindow')

conv_1 = Conv1D( 64 , 19,  strides=1, padding='same', activation='relu', use_bias=True,kernel_regularizer=w_reg, name='Network1-filter1')(conv1_input)
conv_1 = BatchNormalization(name='BN1')(conv_1)
conv_2 = Conv1D( 64 , 11,  strides=1, padding='same', activation='relu', use_bias=True,kernel_regularizer=w_reg, name='Network1-filter2')(conv1_input)
conv_2 = BatchNormalization(name='BN2')(conv_2)
conv_3 = Conv1D( 64 , 3,  strides=1, padding='same', activation='relu', use_bias=True,kernel_regularizer=w_reg, name='Network1-filter3')(conv1_input)
conv_3 = BatchNormalization(name='BN3')(conv_3)

merge_1 = concatenate([conv_1, conv_2, conv_3], name='Network1')
input_for_second = concatenate([conv1_input, merge_1], name='Network1-and-input')



conv_4 = Conv1D( 64 , 19,  strides=1, padding='same', activation='relu', use_bias=True,kernel_regularizer=w_reg, name='Network2-filter1')(input_for_second)
conv_4 = BatchNormalization(name='BN4')(conv_4)
conv_5 = Conv1D( 64 , 11,  strides=1, padding='same', activation='relu', use_bias=True,kernel_regularizer=w_reg, name='Network2-filter2')(input_for_second)
conv_5 = BatchNormalization(name='BN5')(conv_5)
conv_6 = Conv1D( 64 , 3,  strides=1, padding='same', activation='relu', use_bias=True,kernel_regularizer=w_reg, name='Network2-filter3')(input_for_second)
conv_6 = BatchNormalization(name='BN6')(conv_6)

merge_2 = concatenate([conv_4, conv_5, conv_6], name='Network2')
input_for_third = concatenate([conv1_input, merge_1, merge_2],name='Network1-Network2-and-input')



conv_7 = Conv1D( 64 , 19,  strides=1, padding='same', activation='relu', use_bias=True,kernel_regularizer=w_reg, name='Network3-filter1')(input_for_third)
conv_7 = BatchNormalization(name='BN7')(conv_7)
conv_8 = Conv1D( 64 , 11,  strides=1, padding='same', activation='relu', use_bias=True,kernel_regularizer=w_reg, name='Network3-filter2')(input_for_third)
conv_8 = BatchNormalization(name='BN8')(conv_8)
conv_9 = Conv1D( 64 , 3,  strides=1, padding='same', activation='relu', use_bias=True,kernel_regularizer=w_reg, name='Network3-filter3')(input_for_third)
conv_9 = BatchNormalization(name='BN9')(conv_9)

merge_3 = concatenate([conv_7, conv_8, conv_9],name='Network3')

merge_final = concatenate([merge_1, merge_2, merge_3], name='Final')



flatten  = Flatten()(merge_final)
first_dense = Dense(200, activation='relu', use_bias=True,  kernel_regularizer=w_reg, name='last')(flatten)
first_dense = BatchNormalization(name='BN10')(first_dense)
final_model_output = Dense(num_classes, activation = 'softmax', name='softmax')(first_dense)

m = Model(inputs=conv1_input, outputs=final_model_output)

opt = Adam(lr=LR)
m.compile(optimizer=opt, loss=loss,metrics=['accuracy', 'mae'])

print("\nHyper Parameters\n")
print("Learning Rate: " + str(LR))
print("Drop out: " + str(drop_out))
print("Batch dim: " + str(batch_dim))
print("Number of epochs: " + str(nn_epochs))
print("Regularizers: " + str(w_reg.l2))
print("\nLoss: " + loss + "\n")
m.summary()

#import os
#os.environ["PATH"] += os.pathsep + 'C:/Users/Ieremie/Anaconda3/pkgs/graphviz-2.38-hfd603c8_2/Library/bin'

plot_model(m)#, to_file='model.png')







##################




from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv1D, AveragePooling1D, MaxPooling1D, TimeDistributed, LeakyReLU, BatchNormalization, Flatten
from keras import optimizers, callbacks
from keras.regularizers import l2
# import keras.backend as K
import tensorflow as tf



nn_epochs = 25
LR = 0.0009 # maybe after some (10-15) epochs reduce it to 0.0008-0.0007
drop_out = 0.22
batch_dim = 64

loss = 'categorical_crossentropy'

conv1_input = Input(shape=(windowSize, 21), name='InputWindow')
conv_1 = Conv1D( 64, 3,  strides=1, padding='same', activation='relu', use_bias=True)(conv1_input)
conv_1 = BatchNormalization(name='BN1')(conv_1)
conv_1 = Dropout(drop_out)(conv_1)

conv_2 = Conv1D( 64, 3, strides=1, padding='same', activation='relu', use_bias=True)(conv_1)
conv_2 = BatchNormalization(name='BN2')(conv_2)
conv_2 = Dropout(drop_out)(conv_2)

conv_3 = Conv1D( 64, 3,  strides=1, padding='same', activation='relu', use_bias=True)(conv_2)
conv_3 = BatchNormalization(name='BN3')(conv_3)
conv_3 = Dropout(drop_out)(conv_3)

conv_4 = Conv1D( 64, 3, strides=1, padding='same', activation='relu', use_bias=True)(conv_3)
conv_4 = BatchNormalization(name='BN4')(conv_4)
conv_4 = Dropout(drop_out)(conv_4)

conv_5 = Conv1D( 64, 3,  strides=1, padding='same', activation='relu', use_bias=True)(conv_4)
conv_5 = BatchNormalization(name='BN5')(conv_5)
conv_5 = Dropout(drop_out)(conv_5)


input_for_second = concatenate([conv1_input, conv_5], name='Network1-and-input')



conv_6 = Conv1D( 64, 3,  strides=1, padding='same', activation='relu', use_bias=True)(input_for_second)
conv_6 = BatchNormalization(name='BN6')(conv_6)
conv_6 = Dropout(drop_out)(conv_6)

conv_7 = Conv1D( 64, 3, strides=1, padding='same', activation='relu', use_bias=True)(conv_6)
conv_7 = BatchNormalization(name='BN7')(conv_7)
conv_7 = Dropout(drop_out)(conv_7)

conv_8 = Conv1D( 64, 3,  strides=1, padding='same', activation='relu', use_bias=True)(conv_7)
conv_8 = BatchNormalization(name='BN8')(conv_8)
conv_8 = Dropout(drop_out)(conv_8)

conv_9 = Conv1D( 64, 3, strides=1, padding='same', activation='relu', use_bias=True)(conv_8)
conv_9 = BatchNormalization(name='BN9')(conv_9)
conv_9 = Dropout(drop_out)(conv_9)

conv_10 = Conv1D( 64, 3,  strides=1, padding='same', activation='relu', use_bias=True)(conv_9)
conv_10 = BatchNormalization(name='BN10')(conv_10)
conv_10 = Dropout(drop_out)(conv_10)


input_for_third = concatenate([conv1_input, conv_5, conv_10],name='Network1-Network2-and-input')



conv_11 = Conv1D( 64, 3,  strides=1, padding='same', activation='relu', use_bias=True)(input_for_third)
conv_11 = BatchNormalization(name='BN11')(conv_11)
conv_11 = Dropout(drop_out)(conv_11)

conv_12 = Conv1D( 64, 3, strides=1, padding='same', activation='relu', use_bias=True)(conv_11)
conv_12 = BatchNormalization(name='BN12')(conv_12)
conv_12 = Dropout(drop_out)(conv_12)

conv_13 = Conv1D( 64, 3,  strides=1, padding='same', activation='relu', use_bias=True)(conv_12)
conv_13 = BatchNormalization(name='BN13')(conv_13)
conv_14 = Dropout(drop_out)(conv_13)

conv_14 = Conv1D( 64, 3, strides=1, padding='same', activation='relu', use_bias=True)(conv_13)
conv_14 = BatchNormalization(name='BN14')(conv_14)
conv_14 = Dropout(drop_out)(conv_14)

conv_15 = Conv1D( 64, 3,  strides=1, padding='same', activation='relu', use_bias=True)(conv_14)
conv_15 = BatchNormalization(name='BN15')(conv_15)
conv_15 = Dropout(drop_out)(conv_15)



merge_final = concatenate([conv_5, conv_10, conv_15], name='Final')


flatten  = Flatten()(merge_final)
first_dense = Dense(256, activation='relu', use_bias=True)(flatten)
second_dense = Dense(64, activation='relu', use_bias=True)(first_dense)

final_model_output = Dense(classSize, activation = 'softmax', name='softmax')(second_dense)

m = Model(inputs=conv1_input, outputs=final_model_output)

opt = Adam(lr=LR)
m.compile(optimizer=opt, loss=loss,metrics=['accuracy', 'mae'])

print("\nHyper Parameters\n")
print("Learning Rate: " + str(LR))
print("Drop out: " + str(drop_out))
print("Batch dim: " + str(batch_dim))
print("Number of epochs: " + str(nn_epochs))
#print("Regularizers: " + str(w_reg.l2))
print("\nLoss: " + loss + "\n")
m.summary()

#import os
#os.environ["PATH"] += os.pathsep + 'C:/Users/Ieremie/Anaconda3/pkgs/graphviz-2.38-hfd603c8_2/Library/bin'

#plot_model(m)#, to_file='model.png')






#################################


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv1D, AveragePooling1D, MaxPooling1D, TimeDistributed, LeakyReLU, BatchNormalization, Flatten
from keras import optimizers, callbacks
from keras.regularizers import l2
# import keras.backend as K
import tensorflow as tf



nn_epochs = 35
LR = 0.0009 # maybe after some (10-15) epochs reduce it to 0.0008-0.0007
drop_out = 0.3
batch_dim = 64

loss = 'categorical_crossentropy'

conv1_input = Input(shape=(windowSize, 21), name='InputWindow')
conv_1 = Conv1D( 64, 3,  strides=1, padding='same', activation='relu', use_bias=True)(conv1_input)
conv_1 = BatchNormalization(name='BN1')(conv_1)
conv_1 = Dropout(drop_out)(conv_1)

conv_2 = Conv1D( 64, 3, strides=1, padding='same', activation='relu', use_bias=True)(conv_1)
conv_2 = BatchNormalization(name='BN2')(conv_2)
conv_2 = Dropout(drop_out)(conv_2)

conv_3 = Conv1D( 64, 3,  strides=1, padding='same', activation='relu', use_bias=True)(concatenate([conv1_input, conv_2]))
conv_3 = BatchNormalization(name='BN3')(conv_3)
conv_3 = Dropout(drop_out)(conv_3)

conv_4 = Conv1D( 64, 3, strides=1, padding='same', activation='relu', use_bias=True)(conv_3)
conv_4 = BatchNormalization(name='BN4')(conv_4)
conv_4 = Dropout(drop_out)(conv_4)

conv_5 = Conv1D( 64, 3,  strides=1, padding='same', activation='relu', use_bias=True)(concatenate([conv1_input, conv_4]))
conv_5 = BatchNormalization(name='BN5')(conv_5)
conv_5 = Dropout(drop_out)(conv_5)




merge_final = concatenate([conv_1, conv_3, conv_5], name='Final')


flatten  = Flatten()(merge_final)
first_dense = Dense(128, activation='relu', use_bias=True)(flatten)
second_dense = Dense(32, activation='relu', use_bias=True)(first_dense)

final_model_output = Dense(classSize, activation = 'softmax', name='softmax')(second_dense)

m = Model(inputs=conv1_input, outputs=final_model_output)

opt = Adam(lr=LR)
m.compile(optimizer=opt, loss=loss,metrics=['accuracy', 'mae'])

print("\nHyper Parameters\n")
print("Learning Rate: " + str(LR))
print("Drop out: " + str(drop_out))
print("Batch dim: " + str(batch_dim))
print("Number of epochs: " + str(nn_epochs))
#print("Regularizers: " + str(w_reg.l2))
print("\nLoss: " + loss + "\n")
m.summary()

#import os
#os.environ["PATH"] += os.pathsep + 'C:/Users/Ieremie/Anaconda3/pkgs/graphviz-2.38-hfd603c8_2/Library/bin'

#plot_model(m)#, to_file='model.png')