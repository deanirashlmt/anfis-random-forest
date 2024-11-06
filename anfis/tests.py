import anfis
import membership.mfDerivs
import membership.membershipfunction
import numpy as np
import pandas as pd

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

# Pilih fitur dan target yang relevan
# Misalnya, menggunakan 'age' dan 'ejection_fraction' sebagai fitur, dan 'DEATH_EVENT' sebagai target
# X = df.drop(columns=['DEATH_EVENT']).values  # Menghapus kolom target dari input
X = df[['age', 'ejection_fraction']].values

Y = df['DEATH_EVENT'].values  # Kolom target

# Definisikan fungsi keanggotaan
mf = [
    # Age
    [['gaussmf', {'mean': 60, 'sigma': 10}], 
     ['gaussmf', {'mean': 50, 'sigma': 15}], 
     ['gaussmf', {'mean': 40, 'sigma': 10}], 
     ['gaussmf', {'mean': 30, 'sigma': 5}]],

    # Blood Pressure
    [['gaussmf', {'mean': 130, 'sigma': 10}], 
     ['gaussmf', {'mean': 120, 'sigma': 15}], 
     ['gaussmf', {'mean': 110, 'sigma': 10}], 
     ['gaussmf', {'mean': 100, 'sigma': 5}]],

    # Cholesterol
    # [['gaussmf', {'mean': 200, 'sigma': 20}], 
    #  ['gaussmf', {'mean': 180, 'sigma': 25}], 
    #  ['gaussmf', {'mean': 160, 'sigma': 20}], 
    #  ['gaussmf', {'mean': 140, 'sigma': 15}]],

    # # Heart Rate
    # [['gaussmf', {'mean': 70, 'sigma': 5}], 
    #  ['gaussmf', {'mean': 65, 'sigma': 10}], 
    #  ['gaussmf', {'mean': 60, 'sigma': 5}], 
    #  ['gaussmf', {'mean': 55, 'sigma': 5}]],

    # # Creatinine Level
    # [['gaussmf', {'mean': 1.0, 'sigma': 0.1}], 
    #  ['gaussmf', {'mean': 1.2, 'sigma': 0.2}], 
    #  ['gaussmf', {'mean': 0.8, 'sigma': 0.1}], 
    #  ['gaussmf', {'mean': 0.5, 'sigma': 0.1}]],

    # # Ejection Fraction
    # [['gaussmf', {'mean': 60, 'sigma': 10}], 
    #  ['gaussmf', {'mean': 55, 'sigma': 5}], 
    #  ['gaussmf', {'mean': 50, 'sigma': 5}], 
    #  ['gaussmf', {'mean': 45, 'sigma': 5}]],

    # # Sodium Level
    # [['gaussmf', {'mean': 140, 'sigma': 5}], 
    #  ['gaussmf', {'mean': 135, 'sigma': 5}], 
    #  ['gaussmf', {'mean': 130, 'sigma': 5}], 
    #  ['gaussmf', {'mean': 125, 'sigma': 5}]],

    # # Potassium Level
    # [['gaussmf', {'mean': 4.0, 'sigma': 0.1}], 
    #  ['gaussmf', {'mean': 3.8, 'sigma': 0.1}], 
    #  ['gaussmf', {'mean': 3.6, 'sigma': 0.1}], 
    #  ['gaussmf', {'mean': 3.4, 'sigma': 0.1}]],

    # # Blood Sugar Level
    # [['gaussmf', {'mean': 100, 'sigma': 10}], 
    #  ['gaussmf', {'mean': 90, 'sigma': 10}], 
    #  ['gaussmf', {'mean': 80, 'sigma': 10}], 
    #  ['gaussmf', {'mean': 70, 'sigma': 10}]],

    # # Diabetes (Binary: Yes/No)
    # [['gaussmf', {'mean': 1, 'sigma': 0.5}], 
    #  ['gaussmf', {'mean': 0, 'sigma': 0.5}]],  # 1 for Yes, 0 for No

    # # Smoking Status (Binary: Yes/No)
    # [['gaussmf', {'mean': 1, 'sigma': 0.5}], 
    #  ['gaussmf', {'mean': 0, 'sigma': 0.5}]],  # 1 for Yes, 0 for No

    # # Heart Failure History (Binary: Yes/No)
    # [['gaussmf', {'mean': 1, 'sigma': 0.5}], 
    #  ['gaussmf', {'mean': 0, 'sigma': 0.5}]]   # 1 for Yes, 0 for No
]



mfc = membership.membershipfunction.MemFuncs(mf)
anf = anfis.ANFIS(X, Y, mfc)
anf.trainHybridJangOffLine(epochs=20)
print(round(anf.consequents[-1][0],6))
print(round(anf.consequents[-2][0],6))
print(round(anf.fittedValues[9][0],6))
if round(anf.consequents[-1][0],6) == -5.275538 and round(anf.consequents[-2][0],6) == -1.990703 and round(anf.fittedValues[9][0],6) == 0.002249:
	print('test is good')

print("Plotting errors")
anf.plotErrors()
print("Plotting results")
anf.plotResults()

Y_true = Y.flatten() 
Y_pred = anf.fittedValues.flatten()  # Pastikan ini juga 1D

# Tentukan threshold untuk klasifikasi jika diperlukan
threshold = 0.5  # Misalnya, jika hasil prediksi > 0.5 dianggap positif

# Buat prediksi biner
Y_pred_binary = (Y_pred > threshold).astype(int)  # Mengkonversi prediksi menjadi biner

# Hitung jumlah prediksi yang benar
correct_predictions = np.sum(Y_pred_binary == Y_true)

# Hitung total jumlah data
total_data = Y_true.shape[0]

# Hitung akurasi
accuracy = (correct_predictions / total_data) * 100

print(f'Akurasi: {accuracy:.2f}%')