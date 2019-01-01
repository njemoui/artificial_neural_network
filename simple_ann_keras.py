from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
from train_test_data import Data
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np

class ANN:
    def __init__(self):
        self.data = Data("data/churn_modelling.csv")
        self.ann_classifier = self.build_classifier()

    def build_classifier(self,adding_dropout=True):
        ann_classifier = Sequential()
        ann_classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
        if adding_dropout:
            ann_classifier.add(Dropout(p=0.1))
        ann_classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
        if adding_dropout:
            ann_classifier.add(Dropout(p=0.1))
        ann_classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
        ann_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return ann_classifier

    def train_the_ann(self):
        self.ann_classifier.fit(self.data.observations_train, self.data.labels_train, batch_size=10, epochs=100)

    def confusion_matrix(self):
        y_pred = self.ann_classifier.predict(self.data.observations_test)
        y_pred = (y_pred > 0.5)
        cm = confusion_matrix(self.data.labels_test, y_pred)
        print(cm)

    def predict_new_observation(self,observation):
        new_prediction = self.ann_classifier.predict(self.data.stander_scaler.transform(np.array([observation])))
        new_prediction = (new_prediction > 0.5)
        print(new_prediction)

    def tensorflow_memory_usage(self):
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        set_session(sess)

    def evaluating_the_ann(self):
        self.ann_classifier = KerasClassifier(build_fn=self.build_classifier, batch_size=10, epochs=100)
        accuracies = cross_val_score(estimator=self.ann_classifier, X=self.data.observations_train, y=self.data.labels_train, cv=10, n_jobs=1)
        mean = accuracies.mean()
        variance = accuracies.std()
        print("mean : " + str(mean))
        print("variance : " + str(variance))

    def tuning_the_ann(self):
        classifier = KerasClassifier(build_fn=self.build_classifier)
        parameters = {'batch_size': [25, 32],'epochs': [100, 500],'optimizer': ['adam', 'rmsprop']}
        grid_search = GridSearchCV(estimator=classifier,param_grid = parameters,scoring = 'accuracy',cv = 10,n_jobs=1)
        grid_search = grid_search.fit(self.data.observations_train, self.data.labels_train)
        best_parameters = grid_search.best_params_
        best_accuracy = grid_search.best_score_
        print("best_parameters : " + str(best_parameters))
        print("best_accuracy : " + str(best_accuracy))



if __name__ == "__main__":
    ann = ANN()
    ann.tensorflow_memory_usage()
    ann.train_the_ann()
    ann.confusion_matrix()
    ann.predict_new_observation([0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000])
    # ann.evaluating_the_ann()
    # ann.tuning_the_ann()

