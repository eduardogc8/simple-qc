from keras import Sequential, Input, Model
from keras.layers import LSTM, Dropout, Dense, Embedding, Reshape, Conv2D, MaxPool2D, Concatenate, Flatten
from keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


def lstm_default(in_dim=300, out_dim=7, drop=0.2):
    model = Sequential()
    model.add(LSTM(256, input_dim=in_dim, name='0_LSTM'))
    model.add(Dropout(drop, name='1_Droupout'))
    model.add(Dense(128, activation='relu', name='2_Dense'))
    model.add(Dropout(drop, name='3_Droupout'))
    model.add(Dense(out_dim, activation='softmax', name='4_Dense'))
    #otimizer = keras.optimizers.Adam(lr=0.01) #decay = 0.0001
    #model.compile(optimizer=otimizer, loss='categorical_crossentropy')
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy')
    return model


def svm_linear():
    svc = LinearSVC(C=1.0)
    return svc


def random_forest():
    return RandomForestClassifier(n_estimators=500)


def mlp(in_dim=5000, out_dim=7, drop=0.65):
    model = Sequential()
    model.add(Dense(128, input_dim=in_dim, activation='relu'))
    model.add(Dropout(drop))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(drop))
    model.add(Dense(out_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def cnn(sequence_length, vocabulary_size, embedding_dim=300, filter_sizes=[3,4,5], num_filters=512, drop=0.5, out_dim=7):
    inputs = Input(shape=(sequence_length,), dtype='int32')
    embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
    reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=out_dim, activation='softmax')(dropout)

    model = Model(inputs=inputs, outputs=output)
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy')

    return model