from skfuzzy import gaussmf, gbellmf, sigmf

class MemFuncs:
    funcDict = {'gaussmf': gaussmf, 'gbellmf': gbellmf, 'sigmf': sigmf}

    def __init__(self, MFList):
        self.MFList = MFList

    def evaluateMF(self, rowInput):
        # print('Evaluating MF for input row:', rowInput)
        # print("Shape of rowInput:", len(rowInput))
        # print("Number of MF sets:", len(self.MFList))

        if len(rowInput) != len(self.MFList):
            raise ValueError("Number of variables does not match number of rule sets")

        results = []
        for i in range(len(rowInput)):
            mf_results = []
            for k in range(len(self.MFList[i])):
                mf_type = self.MFList[i][k][0]
                mf_params = self.MFList[i][k][1]
                result = self.funcDict[mf_type](rowInput[i], **mf_params)
                mf_results.append(result)
                # print(f"Evaluated {mf_type} MF with params {mf_params} on input {rowInput[i]}: result = {result}")
            results.append(mf_results)
        return results

