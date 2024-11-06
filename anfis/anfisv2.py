# -*- coding: utf-8 -*-
"""
Created on Thu Apr 03 07:30:34 2014

@author: tim.meggs
"""
import itertools
import numpy as np
from membership import mfDerivs
import copy
from skfuzzy import gaussmf, gbellmf, sigmf

class MemFuncs:
    def __init__(self, MFList):
        self.MFList = MFList
        self.funcDict = {'gaussmf': gaussmf, 'gbellmf': gbellmf, 'sigmf': sigmf}

    def evaluateMF(self, rowInput):
        if len(rowInput) != len(self.MFList):
            raise ValueError("Number of variables does not match number of rule sets")
        return [
            [
                self.funcDict[self.MFList[i][k][0]](rowInput[i], **self.MFList[i][k][1]) 
                for k in range(len(self.MFList[i]))
            ] 
            for i in range(len(rowInput))
        ]

class ANFIS:
    def __init__(self, X, Y, memFunction):
        self.X = np.array(copy.copy(X))
        self.Y = np.array(copy.copy(Y))
        self.XLen = len(self.X)
        self.memClass = copy.deepcopy(memFunction)
        self.memFuncs = self.memClass.MFList
        self.memFuncsByVariable = [[x for x in range(len(self.memFuncs[z]))] for z in range(len(self.memFuncs))]
        self.rules = np.array(list(itertools.product(*self.memFuncsByVariable)))
        self.consequents = np.zeros(self.Y.ndim * len(self.rules) * (self.X.shape[1] + 1))
        self.errors = np.empty(0)
        self.memFuncsHomo = all(len(i) == len(self.memFuncsByVariable[0]) for i in self.memFuncsByVariable)
        self.trainingType = 'Not trained yet'

    def LSE(self, A, B, initialGamma=1000.):
        coeffMat = A
        rhsMat = B
        S = np.eye(coeffMat.shape[1]) * initialGamma
        x = np.zeros((coeffMat.shape[1], 1))
        for i in range(len(coeffMat[:, 0])):
            a = coeffMat[i, :]
            b = np.array(rhsMat[i])
            S = S - (np.dot(np.dot(S, np.outer(a, a)), S)) / (1 + np.dot(a, S @ a))
            x = x + (S @ np.dot(a, (b - np.dot(a, x))))
        return x


    def trainHybridJangOffLine(self, epochs=5, tolerance=1e-5, initialGamma=1000, k=0.01):
        self.trainingType = 'trainHybridJangOffLine'
        convergence = False
        epoch = 1

        while (epoch < epochs) and (not convergence):
            print(f"\nEpoch {epoch}/{epochs} - Training started.")

            # layer empat: forward pass
            layerFour, wSum, w = self.forwardHalfPass(self.X)

            # layer lima: least squares estimate
            layerFive = np.array(self.LSE(layerFour, self.Y, initialGamma))
            self.consequents = layerFive
            layerFive = np.dot(layerFour, layerFive)

            # Menghitung error
            error = np.sum((self.Y - layerFive.T) ** 2)
            print(f"Epoch {epoch} - Current Error: {error}")
            average_error = np.average(np.absolute(self.Y - layerFive.T))
            self.errors = np.append(self.errors, error)

            # Cek toleransi error untuk konvergensi
            if len(self.errors) != 0 and self.errors[-1] < tolerance:
                convergence = True
                print("Converged successfully.")

            # Backpropagation jika belum konvergen
            if not convergence:
                print(f"Epoch {epoch} - Running backpropagation...")
                cols = range(len(self.X[0, :]))
                dE_dAlpha = [self.backprop(self, colX, cols, wSum, w, layerFive) for colX in range(self.X.shape[1])]

                print(f"Epoch {epoch} - Completed backpropagation.")

            # Update epoch
            epoch += 1

        print("\nTraining completed.")
        self.fittedValues = self.predict(self.X)
        self.residuals = self.Y - self.fittedValues[:, 0]

        return self.fittedValues

    def plotErrors(self):
        if self.trainingType == 'Not trained yet':
            print(self.trainingType)
        else:
            import matplotlib.pyplot as plt
            plt.plot(range(len(self.errors)),self.errors,'ro', label='errors')
            plt.ylabel('error')
            plt.xlabel('epoch')
            plt.show()

    def plotMF(self, x, inputVar):
        import matplotlib.pyplot as plt
        from skfuzzy import gaussmf, gbellmf, sigmf

        for mf in range(len(self.memFuncs[inputVar])):
            if self.memFuncs[inputVar][mf][0] == 'gaussmf':
                y = gaussmf(x,**self.memClass.MFList[inputVar][mf][1])
            elif self.memFuncs[inputVar][mf][0] == 'gbellmf':
                y = gbellmf(x,**self.memClass.MFList[inputVar][mf][1])
            elif self.memFuncs[inputVar][mf][0] == 'sigmf':
                y = sigmf(x,**self.memClass.MFList[inputVar][mf][1])

            plt.plot(x,y,'r')

        plt.show()

    def plotResults(self):
        if self.trainingType == 'Not trained yet':
            print(self.trainingType)
        else:
            import matplotlib.pyplot as plt
            plt.plot(range(len(self.fittedValues)),self.fittedValues,'r', label='trained')
            plt.plot(range(len(self.Y)),self.Y,'b', label='original')
            plt.legend(loc='upper left')
            plt.show()


    def forwardHalfPass(self, Xs):
        layerFour = np.empty(0,)
        wSum = []
        w = None  # Inisialisasi w sebelum digunakan

        for pattern in range(len(Xs[:, 0])):
            # Menghitung layer satu
            layerOne = self.memClass.evaluateMF(Xs[pattern, :])

            # Menghitung layer dua
            miAlloc = [
                [layerOne[x][self.rules[row][x]] for x in range(len(self.rules[0]))]
                for row in range(len(self.rules))
            ]
            layerTwo = np.array([np.prod(x) for x in miAlloc]).T

            # Menyimpan bobot
            w = layerTwo if w is None else np.vstack((w, layerTwo))

            # Menghitung jumlah bobot
            wSum.append(np.sum(layerTwo))

            # Normalisasi bobot dengan pemeriksaan untuk menghindari pembagian dengan nol
            if wSum[pattern] != 0:
                wNormalized = layerTwo / wSum[pattern]
            else:
                wNormalized = np.zeros_like(layerTwo)  # Nilai default jika wSum adalah nol

            # Menciptakan rowHolder
            rowHolder = np.concatenate([x * np.append(Xs[pattern, :], 1) for x in wNormalized])
            layerFour = np.append(layerFour, rowHolder)

        w = w.T
        layerFour = np.array(np.array_split(layerFour, pattern + 1))

        return layerFour, wSum, w


    def backprop(self, columnX, columns, theWSum, theW, theLayerFive):
        paramGrp = [0] * len(self.memFuncs[columnX])
        for MF in range(len(self.memFuncs[columnX])):
            parameters = np.empty(len(self.memFuncs[columnX][MF][1]))
            for timesThru, alpha in enumerate(sorted(self.memFuncs[columnX][MF][1].keys())):
                bucket3 = np.empty(len(self.X))
                for rowX in range(len(self.X)):
                    varToTest = self.X[rowX, columnX]
                    tmpRow = np.empty(len(self.memFuncs))
                    tmpRow.fill(varToTest)

                    bucket2 = np.empty(self.Y.ndim)
                    rulesWithAlpha = np.where(self.rules[:, columnX] == MF)[0]
                    adjCols = np.delete(columns, columnX)
                    senSit = mfDerivs.partial_dMF(self.X[rowX, columnX], self.memFuncs[columnX][MF], alpha)
                    dW_dAlpha = senSit * np.array([
                        np.prod([self.memClass.evaluateMF(tmpRow)[c][self.rules[r][c]] for c in adjCols]) for r in rulesWithAlpha
                    ])

                    bucket1 = np.empty(len(self.rules[:, 0]))
                    for consequent in range(len(self.rules[:, 0])):
                        fConsequent = np.dot(np.append(self.X[rowX, :], 1.), self.consequents[((self.X.shape[1] + 1) * consequent):(((self.X.shape[1] + 1) * consequent) + (self.X.shape[1] + 1))])
                        acum = (dW_dAlpha[np.where(rulesWithAlpha == consequent)] * theWSum[rowX] if consequent in rulesWithAlpha else 0) - theW[consequent, rowX] * np.sum(dW_dAlpha)
                        acum /= theWSum[rowX] ** 2
                        bucket1[consequent] = fConsequent * acum

                    sum1 = np.sum(bucket1)
                    bucket2[0] = sum1 * (self.Y[rowX] - theLayerFive[rowX]) * -2
                    sum2 = np.sum(bucket2)
                    bucket3[rowX] = sum2

                parameters[timesThru] = np.sum(bucket3)
            paramGrp[MF] = parameters
        return paramGrp

    def predict(self, varsToTest):
        layerFour, wSum, w = self.forwardHalfPass(varsToTest)
        return np.dot(layerFour, self.consequents)

# Example usage:
if __name__ == "__main__":
    print("This is the main ANFIS module!")
