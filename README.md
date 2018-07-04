##### Claudia Espinoza<br>410321168
# Character Recognition




## Dimensionality Reduction
Find an eigenvector for each matrix 28x28. <br>
Total Eigenvectors = Total Pictures<br>
The dimensionality of each eigenvector is 28*28 = 784.<br>
**explained variance:** tells us how much information can be attributed to each of the principal components<br>
  After several trials **300** principal components were kept. The variance retained in the trials was as follows:<br>
  * **100** PC needed for **cumulative variance = .74** with **svd_solver = full**<br>
  * **300** PC needed for **cumulative variance = .95** with **svd_solver = auto**<br>
  * **300** PC needed for **cumulative variance = .97** with **svd_solver = full**<br>

  ```python
  def PCAnalysis(train, test):
      pca = PCA(300, svd_solver='full') #choose 300 after several trials
      train = pca.fit_transform(train)
      test = pca.transform(test)
      pca_std = np.std(train)

      #Show variance graph to choose an adequate number of components to keep
      print("VARIANCE RATIO: " + str(pca.explained_variance_ratio_.cumsum()))
      print("PCA SUCCESSFUL")

      return (train, test, pca_std)
  ```

## Preprocessing
Standardize features by removing the mean and scaling to unit variance.
```python
scaler = StandardScaler()
scaler.fit(trainingset)
trainingset = scaler.transform(trainingset)
testset = scaler.transform(testset)
```
***Accuracy remains under .41 if we do not standarize the mean and variation of the dataset.***

## Training Models
#### Keras NN
###### training accuracy: .99<br>validation accuracy: .97<br>
**Dense Layer:** take as input arrays of shape (samples, x) and output arrays of shape (samples, y)<br>
**Gaussian Noise:** reduce overfitting<br>
```python
def LearnKeras(train, ytrain, testset, ytest, pca_std):
    model = Sequential()
    layers = 1
    units = 128
    batch = 60
    ytrain = to_categorical(ytrain)
    ytest = to_categorical(ytest)

    model.add(Dense(units, input_dim=300, activation='relu'))
    model.add(GaussianNoise(pca_std))
    for i in range(layers):
        model.add(Dense(units, activation='relu'))
        model.add(GaussianNoise(pca_std))
        model.add(Dropout(0.1))
    model.add(Dense(10, activation='softmax'))

 	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])
 	model.fit(train, ytrain, epochs=100, batch_size=batch, validation_data=(testset,ytest))
```

#### KNN with sklearn
```python
def LearnKNN(train, ytrain, test, ytest):
    print("KNN Classifier")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train, ytrain)
    predicted = knn.predict(test)
    acc = accuracy_score(ytest, predicted)
    print(acc)
    print("done")
```
## Conclusion
It is very important to standarize the data before further processing. Look for a suitable number of principal components to retain when doing PCA, by comparing how much variance each PC retains. If the variance value after a certain component is not significant we do not need to keep adding more components. If the data is prepocessed properly, the difference in accuracy between model doesnt vary much.<br>
###### KNN-> time: 11:55:30 accuracy: 0.9506<br>KERAS Model-> time: 11:55:30 accuracy: 0.97

##### Comments
* Unable to download data because the page was down. Had to download datasets and extract pixel values with function.
* Prints and further especifications were ommitted in the code blocks of the README, to see full source code, check **char_recog.py**
