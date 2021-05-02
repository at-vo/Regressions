'''
        DAT VO
        250983323
        CS 4442 ASN 1
        BOYU WANG
'''

import matplotlib.pyplot as plt
import numpy as np
from numpy.core.numeric import identity
import pandas as pd
from pandas.core.indexing import is_label_like

# PATH variables for data
pathXTR = "hw1xtr.dat"
pathYTR = "hw1ytr.dat"
pathXTE = "hw1xte.dat"
pathYTE = "hw1yte.dat"

# Str list for printing order of a function
degreeList = ["Linear", "Second Order", "Third Order", "Fourth Order"]

def readfile(predict:str, expect:str):
    xdata = np.fromfile(predict, sep=" ")
    ydata = np.fromfile(expect, sep=" ")

    # sort data
    ydata = ydata[np.argsort(xdata)]
    xdata = np.sort(xdata)

    return xdata, ydata

# generate predicted values based on training data and weights
def hypothesis(w, x):
    return w.T @ x

# error function for calculating y1
def errorfn(y1, y):
    return np.sum(pow((y1 - y),2)) / len(y)

# calculate w based on formula
def wCalc(y, lm = 0, i = 0):
    return lambda x: np.linalg.inv((x.T @ x) + (lm * i)) @ x.T @ y

# for easy matrix manipulation
def createDF(xdata, ydata):
    dataobject = {'x':xdata,'y':ydata}
    df = pd.DataFrame(dataobject)
    ydf = df['y']
    xdf = df.drop(columns='y')
    return xdf, ydf

def regression(xtrData, ytrData, xteData, yteData, degree = 1, lm = 0, graph = True):
    # create dataFrames
    xtr, ytr = createDF(xtrData,ytrData)
    xte, yte = createDF(xteData,yteData)

    # add the next order of x values based on degree
    if degree > 1:
        exp = 2
        while exp <= degree:
            xtr[exp] = pow(xtr['x'], exp)
            xte[exp] = pow(xte['x'], exp)
            exp += 1

    # normalize 
    xtr = xtr/xtr.max()
    xte = xte/xte.max()

    # add vector of 1s
    xtr = pd.concat([xtr, pd.Series(1., index = xtr.index, name='bias')], axis=1)
    xte = pd.concat([xte, pd.Series(1., index = xte.index, name='bias')], axis=1)

    # convert to numpy
    xtr = xtr.to_numpy()
    ytr = ytr.to_numpy()
    xte = xte.to_numpy()
    yte = yte.to_numpy()

    # calculate w based on lambda value
    if lm == 0:
        wTr = wCalc(ytr)
    else:
        # add identity matrix, set first entry to 0
        idM = np.identity(degree + 1)
        idM[0, 0] = 0
        wTr = wCalc(ytr, lm, idM)

    # calculate weights on x training 
    weights = wTr(xtr)
    #print(weights)

    # compute y1 values for weights
    y1Tr = [hypothesis(weights, x) for x in xtr]
    y1Te = [hypothesis(weights, x) for x in xte]

    # errors
    errTr = errorfn(y1Tr, ytr)
    errTe = errorfn(y1Te, yte)

    # for Q2
    if graph :
        # graph training 
        fig,axs = plt.subplots(2)
        fig.tight_layout()
        axs[0].scatter(xtrData, ytrData)
        axs[0].set_title(degreeList[degree - 1] + " Regression vs Training Data")
        axs[0].plot(xtrData, y1Tr,color="orange")

        # graph testing
        axs[1].scatter(xteData, yteData)
        axs[1].set_title(degreeList[degree - 1] + " Regression vs Test Data")
        axs[1].plot(xteData, y1Te,color="orange")

        plt.show()

        # report average error
        print("\nRegression Model: ", degreeList[degree - 1])
        print("Average error of Training: ", errTr)
        print("Average error of Testing: ", errTe)

    # return dict of values
    return {"ws": weights, "y1te": y1Te, "errTr": errTr, "errTe": errTe, "order":degree}

def kfold(xtrData, ytrData, folds, degree, lambdalist = [0]):
    # shuffle the data randomly
    xPerm = np.random.permutation(xtrData)
    reorder = [np.where(data == xtrData)[0][0] for data in xPerm]

    # print("\n\n\n\ntraining X",xtrData)
    # for data in range(len(xPerm)):
    #     temp = np.where(xPerm[data] == xtrData)
    #     print("value in permutation:",xPerm[data])
    #     print("index in training: ",temp)
    print(xtrData)
    print(type(xtrData))
    print(reorder)
    print(xtrData[reorder])
    # split the data in k folds
    xSplit = np.split(xtrData[reorder], folds)
    ySplit = np.split(ytrData[reorder], folds)

    errors = []

    for i in range(folds):
        validX = xSplit[i]
        validY = ySplit[i]

        # remove the validX and validY from folds and use as training
        vectorX = np.asarray(xSplit[:i] + xSplit[i+1:]).reshape(-1)
        vectorY = np.asarray(ySplit[:i] + ySplit[i+1:]).reshape(-1)

        # extract testing errors from regression
        errors.append([regression(vectorX, vectorY, validX, validY, 4, lm, False)["errTe"] for lm in lambdalist])

    # calculate average error 
    avgErr = [sum(i)/len(errors) for i in np.array(errors).T]

    return avgErr, lambdalist[avgErr.index(min(avgErr))]

# Check Test Errors
def testError(regressions, test):
    print("\nFor {} error:".format(test))
    minTest = float('inf')
    minTestStr = ""
    tStr = "err" + test[:2]
    for i in range(len(regressions) - 1):    
        # compare regressions
        if regressions[i + 1][tStr] < regressions[i][tStr] :
            low = regressions[i + 1]
            high = regressions[i]
        else:
            low = regressions[i]
            high = regressions[i + 1]
        # compare min errors
        if low[tStr] < minTest:
            minTest = low[tStr]
            minTestStr = degreeList[low["order"] - 1]

        print("{} is better than {}".format(degreeList[low["order"] - 1], degreeList[high["order"] - 1]))
    print("best {} error is {}".format(test, minTestStr))

    return minTestStr

def main():
    # generate arrays for data
    xtrData, ytrData = readfile(pathXTR,pathYTR)
    xteData, yteData = readfile(pathXTE,pathYTE)

    # Plot training and test data
    fig,axs = plt.subplots(2)
    fig.tight_layout()
    axs[0].scatter(xtrData, ytrData)
    axs[0].set_title("Training Data")
    axs[1].scatter(xteData, yteData)
    axs[1].set_title("Test Data")
    plt.show()

    # perform regressions for each order upto 4th
    regressions = [regression(xtrData, ytrData, xteData, yteData, deg, 0, False) for deg in range(1, 5)]
    # for x in regressions:
    #     print(x["ws"])

    # Check Training Errors
    minTrainStr = testError(regressions,"Training")
    minTestStr = testError(regressions,"Testing")

    # Compare Training and Test errors
    print("\n2D:")
    if minTrainStr == minTestStr:
        print("{} is the best for fitting data".format(minTestStr))
    else:
        print("Hard to tell which order is the best since training and test errors do not correlate")

    print("\n3A")

    # Perform Regularization
    lambdaList = [0.01,0.1,1,10,100,1000,10000]
    regularizations = [regression(xtrData, ytrData, xteData, yteData, 4, l, False) for l in lambdaList]
    
    # training and testing errors from regularizations
    lambdaErrorTr = [i["errTr"] for i in regularizations]
    lambdaErrorTe = [i["errTe"] for i in regularizations]

    # min lambda from each training and test
    trLambda = lambdaList[lambdaErrorTr.index(min(lambdaErrorTr))]
    teLambda = lambdaList[lambdaErrorTe.index(min(lambdaErrorTe))]

    # compare lambda errors
    if trLambda == teLambda:
        bestLambda3C = trLambda
        print("training and test error correlate, Best lambda is ", bestLambda3C)
    else:
        print("Best training lambda: ", trLambda)
        print("Best test lambda: ", teLambda)


    # plot training and testing errors as a function of lambda
    plt.plot(lambdaList, lambdaErrorTr, label="Training")
    plt.plot(lambdaList, lambdaErrorTe, label="Test")
    plt.xscale("log")
    plt.title("Errors as a function of lambda")
    plt.xlabel("lambda")
    plt.ylabel("Errors")
    for i,j in zip(lambdaList,lambdaErrorTr):
        plt.annotate(str(j),xy=(i,j))
    plt.legend()
    plt.show()

    # Transpose matrix of Ws to size of lambdalist
    wReg = np.array([i["ws"].tolist() for i in regularizations]).T.tolist()
    
    # plot weights as a function of lambda
    for i in range(len(wReg)):    
        plt.plot(lambdaList, wReg[i], label = "w" + str(i))
    plt.title("Weights vs Lambda")
    plt.xscale("log")
    plt.xlabel("lambda")
    plt.ylabel("weight")
    plt.legend()    
    plt.show()

    k = 5
    # k-fold cross validation for avgError and lambdas
    avgError, bestLambda = kfold(xtrData, ytrData, k, 4, lambdaList)

    # print("\n4C")
    # print("Best lambda is ", bestLambda)
    # if bestLambda == bestLambda3C:
    #     print("Best lambda same as in 3C")

    # compute line of best fit based on best lambda
    lobf = regression(xtrData, ytrData, xteData, yteData, 4, bestLambda, False)

    # Plot Cross Validations
    fig, axs = plt.subplots(2)
    fig.tight_layout()
    axs[0].plot(lambdaList, avgError)
    axs[0].set_title("Cross Validation on lambda")
    axs[0].set_xscale("log")

    axs[1].scatter(xteData, yteData)
    axs[1].plot(xteData, lobf["y1te"], "orange")
    plt.title("Cross Validation on training data")
    plt.show()

    return 0

if __name__ == '__main__':
    main()