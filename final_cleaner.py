import os
import pickle
import pandas as pd
from data_cleaning import SessionData  # Daha önce yazdığın fonksiyonları içe aktar

class F1DataCleaner:
    def __init__(self, folder_path):
        """
        F1 veri temizleyici sınıfı.

        :param folder_path: PKL dosyalarının bulunduğu klasörün yolu.
        """
        self.folder_path = folder_path
        self.cleaned_sessions = []  # Temizlenen verileri saklamak için liste

    def is_valid_session(self, session, filename):
        """
        Yarış verisinin geçerli olup olmadığını kontrol eder.

        :param session: FastF1 oturum verisi.
        :param filename: Dosya adı (hata mesajları için).
        :return: Geçerli ise True, değilse False.
        """
        if session.results is None or session.laps is None:
            print(f"⚠️ {filename}: session.results veya session.laps yüklenemedi, dosya atlanıyor.")
            return False  # Eğer veri yüklenmemişse direkt geçersiz kabul et

        # 🛠 **Position sütunu NaN içeriyorsa dosyayı atla**
        if session.results["Position"].isnull().any():
            print(f"⚠️ {filename}: Position verisi eksik, dosya atlanıyor.")
            return False

        num_drivers_results = session.results["Abbreviation"].nunique()
        num_drivers_laps = session.laps["Driver"].nunique()

        if num_drivers_results != 20 or num_drivers_laps != 20:
            print(f"⚠️ {filename}: 20 sürücü içermiyor, dosya atlanıyor.")
            return False

        return True


    def process_files(self):
        file_list = [f for f in os.listdir(self.folder_path) if f.endswith(".pkl")]

        total_files = len(file_list)
        processed_files = 0  # İşlenen dosya sayacı

        for filename in file_list:
            processed_files += 1
            print(f"Processing {processed_files}/{total_files}: {filename}")  # tqdm yerine basit print

            file_path = os.path.join(self.folder_path, filename)
        
            with open(file_path, "rb") as file:
                session = pickle.load(file)

            

            if not self.is_valid_session(session, filename):
                print(f"⏩ {filename} atlandı (Geçersiz veri)")
                continue  # Geçersiz dosyalar atlanıyor

            session_data = SessionData(session)  
            session_data.clean_data()  
            self.cleaned_sessions.append(session_data)  # Temizlenmiş veriyi listeye ekle

        print(f"\n✅ Toplam {len(self.cleaned_sessions)} yarış başarıyla temizlendi ve listeye eklendi.")








