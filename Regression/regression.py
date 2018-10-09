import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def linearRegression(x, y):
    return np.dot(np.dot(np.linalg.pinv(np.dot(x.transpose(), x)), x.transpose()), y)



def main():
    trainingData = pd.read_csv('housing_training.csv', skiprows=[0], header=None)
    testingData = pd.read_csv('housing_test.csv', skiprows=[0], header=None)

    trainLabels = trainingData.iloc[:, 0]
    testLabels = testingData.iloc[:, 0]
    testData = testingData.iloc[:, 1:]
    trainData = trainingData.iloc[:, 1:]

    trainData = trainData.values
    testData = testData.values

    ones = np.ones((len(trainData), 1))
    one = np.ones((len(testData), 1))

    trainData = np.hstack((trainData, ones))
    testData = np.hstack((testData, one))

    # print(newTrain)

    # print(newTrain.shape, ones.shape)

    b_opt = linearRegression(trainData, trainLabels)
    print(b_opt)

    # print(trainLabels)

    predictions = []
    predictions = np.array(np.dot(testData, b_opt))

    # print(len(testLabels))

    rmse = 0
    # print(predictions[k])
    for k in range(len(testLabels)):
        rmse += np.power(predictions[k] - testLabels[k], 2)
    rmse = rmse / len(testLabels)
    rmse = np.sqrt(rmse)

    print("Root Mean Squared Error:", rmse)

    plt.scatter(predictions, testLabels)
    # plt.scatter(testLabels, testData)
    plt.xlabel("Predictions")
    plt.ylabel("Ground Truth")
    plt.show()


if __name__ == '__main__':
    main()


# Determine residuals : Difference between ground truth and predictions. Basically RMS
