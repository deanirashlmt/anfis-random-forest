import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

# Memuat dataset
df = pd.read_csv("heart_failure_clinical_records_dataset.csv")  # Ganti dengan path ke file CSV Anda

# Menyiapkan fitur dan target
X = df.drop(columns=['DEATH_EVENT'])
y = df['DEATH_EVENT']

# Membagi data menjadi data pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# Menentukan parameter yang ingin dicari
param_grid = {
    'n_estimators': [50, 100, 150],  # Jumlah pohon
    'max_depth': [None, 10, 20, 30],  # Kedalaman maksimum pohon
    'min_samples_split': [2, 5, 10],  # Jumlah minimum sampel yang diperlukan untuk membagi node
    'min_samples_leaf': [1, 2, 4],    # Jumlah minimum sampel yang diperlukan untuk menjadi daun
}

# Membuat model Random Forest
rf = RandomForestClassifier(random_state=42)

# Menggunakan GridSearchCV untuk menemukan kombinasi parameter terbaik dengan K-Fold Cross-Validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           scoring='accuracy', cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Menampilkan hasil
print("Best parameters found: ", grid_search.best_params_)
print("Best accuracy found: ", grid_search.best_score_)

# Menggunakan model terbaik untuk prediksi 
best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(X_test)

# Menghitung akurasi pada data uji
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi pada data uji: {accuracy:.2f}%')
