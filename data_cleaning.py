import fastf1
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

class SessionData:
    def __init__(self, session):
        self.laps = session.laps.copy() if hasattr(session, 'laps') else None
        #self.weather = session.weather_data.copy() if hasattr(session, 'weather_data') else None
        self.results = session.results.copy() if hasattr(session, 'results') else None
        self.session_info = session.session_info.copy() if hasattr(session, 'results') else None
    
    def complete_missing_laps(self):
        if self.laps is None or self.results is None:
            return

        df = self.laps
        max_lap = df['LapNumber'].max()
        drivers = df['Driver'].unique()
        new_rows = []

        for driver in drivers:
            driver_laps = df[df['Driver'] == driver]
            first_record = driver_laps.iloc[0]
            driver_number = first_record['DriverNumber']
            last_record = driver_laps.iloc[-1]
            compound = last_record['Compound']
            team = last_record['Team']
            existing_laps = set(driver_laps['LapNumber'])
            tyre_life =last_record['TyreLife']

            # Eğer sürücü results tablosunda varsa, pozisyon bilgisini al
            result_row = self.results[self.results['DriverNumber'] == driver_number]

            if not result_row.empty:
                final_position = result_row['Position'].iloc[0]
            else:
                final_position = np.nan  # Eğer bilgi yoksa NaN bırak

            for lap in range(1, int(max_lap) + 1):
                if lap not in existing_laps:
                    new_row = {
                        'Driver': driver,
                        'DriverNumber': driver_number,
                        'LapNumber': lap,
                        'Compound': compound,
                        'Team': team,
                        'Position': final_position,
                        'TyreLife': (tyre_life + 1), 
                    }

                    for col in df.columns:
                        if col not in new_row:
                            if col in ['Time', 'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 
                                       'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime', 
                                       'LapStartTime', 'LapStartDate']:
                                new_row[col] = pd.Timedelta(seconds=0)  # "0 days 00:00:00.000000" formatında
                            elif col in ['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST','FreshTyre']:
                                new_row[col] = 0  # Hız sütunlarını 0 ile doldur
                            else:
                                new_row[col] = np.nan  # Geri kalanları NaN bırak

                    new_rows.append(new_row)

        if new_rows:
            df_new = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        else:
            df_new = df.copy()

        self.laps = df_new.sort_values(['Driver', 'LapNumber'])


    
    def drop_columns(self):
        if self.laps is not None:
            self.laps.drop(['IsAccurate', 'LapStartDate', 'FastF1Generated', 'DeletedReason', 'Deleted', 'IsPersonalBest', 'Stint', 'PitOutTime', 'PitInTime', 'TrackStatus'], axis='columns', inplace=True)
    
    def fill_times(self):
        if self.laps is not None:
            grouped = self.laps.groupby('Driver')
            columns_to_fill = ['Time','LapTime','Sector1Time','Sector2Time', 'Sector3Time', 'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime','SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST','LapStartTime']
            self.laps[columns_to_fill] = grouped[columns_to_fill].apply(lambda x: x.bfill().ffill()).reset_index(level=0, drop=True)
    
    def tyre_life(self):
        if self.laps is not None and 'TyreLife' in self.laps.columns:
            self.laps['TyreLife'] = self.laps['TyreLife'].fillna(0.0)

    def fill_position(self):
        if self.laps is not None and 'Position' in self.laps.columns:
            self.laps['Position'] = self.laps.groupby('Driver')['Position'].apply(lambda x: x.bfill().ffill()).reset_index(level=0, drop=True)
            
    def fill_speed(self):
        if self.laps is not None:
            grouped = self.laps.groupby('Driver')
            columns_to_fill = ['SpeedI1', 'SpeedI2', 'SpeedFL','SpeedST']
            self.laps[columns_to_fill] = grouped[columns_to_fill].transform(lambda x: x.bfill().ffill()).reset_index(level=0, drop=True)
    
    def fill_pit_times(self):
        if self.laps is not None:
            self.laps['PitOutTime'] = self.laps['PitOutTime'].fillna(pd.Timedelta(seconds=0))
            self.laps['PitInTime'] = self.laps['PitInTime'].fillna(pd.Timedelta(seconds=0))
    
    def convert_time_columns(self):
        if self.laps is not None:
            time_columns = ['Time', 'LapTime', 'LapStartTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime']
            for col in time_columns:
                if col in self.laps.columns:
                    try:
                        self.laps[col] = pd.to_timedelta(self.laps[col]).dt.total_seconds()
                    except Exception as e:
                        print(f"Error converting column {col}: {e}")
    
    def normalize_columns(self):
        if self.laps is not None:
            scaler = MinMaxScaler()
            columns = ['Time', 'LapTime', 'LapStartTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime','SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'TyreLife']
            for col in columns:
                if col in self.laps.columns:
                    self.laps[col] = scaler.fit_transform(self.laps[[col]])
    
    def clean_data(self):
        self.complete_missing_laps()
        self.drop_columns()
        self.fill_times()
        self.fill_speed()
        self.fill_position()
        self.tyre_life()
        #self.fill_pit_times()
        self.convert_time_columns()
        #self.normalize_columns()
    
    @staticmethod
    def load_from_pkl(file_path):
        with open(file_path, 'rb') as f:
            session = pickle.load(f)
        return SessionData(session)
