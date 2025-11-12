from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import random

class TrainTestData:
    def __init__(self):
        self.train = None
        self.test = None

class LSTMDataPreparer:
    def __init__(self, cleaned_sessions, max_laps=40, test_ratio=0.2, random_state=42):
        self.cleaned_sessions = cleaned_sessions
        self.max_laps = max_laps
        self.num_features = 9  # LapTime, TyreLife, FreshTyre, SpeedST, LapNumber
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.scaler = MinMaxScaler()

    def extract_features(self, driver_laps, fit_scaler=False):
        """Sürücünün lap verisini al, padle ve scale et."""
        features = driver_laps[['LapTime', 'TyreLife', 'FreshTyre', 'SpeedST', 'LapNumber','Position','SpeedI1', 'SpeedI2', 'SpeedFL',]].copy()

        # Padding ya da truncate işlemi
        current_laps = features.shape[0]
        if current_laps < self.max_laps:
            padding = pd.DataFrame(np.zeros((self.max_laps - current_laps, self.num_features)),
                                   columns=features.columns)
            features = pd.concat([features, padding], ignore_index=True)
        else:
            features = features.iloc[:self.max_laps]

        # Ölçekleme
        if fit_scaler:
            self.scaler.partial_fit(features.values)
        else:
            features = pd.DataFrame(self.scaler.transform(features.values), columns=features.columns)

        return features.values  # (max_laps, num_features)

    def extract_from_sessions(self, session_list, fit_scaler=False):
        x_data = []
        y_data = []

        for session in session_list:
            if not hasattr(session, 'laps') or not hasattr(session, 'results'):
                continue
            if session.laps is None or session.results is None:
                continue

            laps = session.laps
            results = session.results

            # 🔁 Abbreviation'a göre sıralayıp tekil eşleşme yapılacak
            results_sorted = results.sort_values("Abbreviation").reset_index(drop=True)

            session_x = []
            session_y = []

            for _, row in results_sorted.iterrows():
                driver_number = row['DriverNumber']
                position = row['Position']
            
                # laps içinden bu sürücünün verisini bul
                driver_laps = laps[laps['DriverNumber'] == driver_number]
                if driver_laps.empty:
                    continue

                features = self.extract_features(driver_laps, fit_scaler=fit_scaler)
                if features.shape != (self.max_laps, self.num_features):
                    continue

                session_x.append(features)
                session_y.append(position - 1)

            # 20 sürücüyü sağlıklı şekilde bulduysak kaydet
            if len(session_x) == 20 and len(session_y) == 20:
                x_data.append(np.array(session_x))
                y_data.append(np.array(session_y))

        return np.array(x_data, dtype=np.float32), np.array(y_data, dtype=np.int32)



    def prepare(self):
        # 1. Shuffle
        sessions = self.cleaned_sessions.copy()
        random.seed(self.random_state)
        random.shuffle(sessions)

        # 2. Train-test ayır
        split_index = int(len(sessions) * (1 - self.test_ratio))
        train_sessions = sessions[:split_index]
        test_sessions = sessions[split_index:]

        # 3. Sadece train verisiyle scaler'ı fit et
        for session in train_sessions:
            if hasattr(session, 'laps') and session.laps is not None:
                for driver_id in session.results['DriverNumber'].unique():
                    driver_laps = session.laps[session.laps['DriverNumber'] == driver_id]
                    self.extract_features(driver_laps, fit_scaler=True)

        # 4. Verileri çıkar
        x_train, y_train = self.extract_from_sessions(train_sessions, fit_scaler=False)
        x_test, y_test = self.extract_from_sessions(test_sessions, fit_scaler=False)

        x = TrainTestData()
        y = TrainTestData()
        x.train, x.test = x_train, x_test
        y.train, y.test = y_train, y_test

        print(f"✅ [Train] X: {x.train.shape}, y: {y.train.shape}")
        print(f"✅ [Test]  X: {x.test.shape}, y: {y.test.shape}")

        return x, y, self.scaler, test_sessions
