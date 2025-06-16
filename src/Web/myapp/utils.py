import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from datetime import datetime
import traceback
import joblib
from sklearn.base import BaseEstimator

def detect_target_column(df):
    """
    Veri setindeki hedef değişkeni akıllı bir şekilde tespit et
    
    Args:
        df: pandas DataFrame
    
    Returns:
        tuple: (hedef_sütun_adı, özellik_sütunları)
    """
    # Veri seti hakkında bilgi topla
    column_info = {}
    for col in df.columns:
        unique_values = df[col].unique()
        n_unique = len(unique_values)
        value_counts = df[col].value_counts()
        
        # Sütun özelliklerini hesapla
        column_info[col] = {
            'n_unique': n_unique,
            'is_binary': n_unique == 2 and set(unique_values).issubset({0, 1, True, False}),
            'is_categorical': n_unique < len(df) * 0.1,  # Benzersiz değer sayısı veri setinin %10'undan azsa kategorik
            'is_numeric': pd.api.types.is_numeric_dtype(df[col]),
            'entropy': -sum((value_counts / len(df)) * np.log2(value_counts / len(df))),  # Shannon entropisi
            'is_last_column': col == df.columns[-1],
            'missing_ratio': df[col].isnull().mean()
        }
    
    # Hedef değişken için puan hesapla
    target_scores = {}
    for col, info in column_info.items():
        score = 0
        
        # Binary sınıflandırma için
        if info['is_binary']:
            score += 5  # Binary değişkenler genellikle hedef değişkendir
        
        # Kategorik değişkenler için
        if info['is_categorical']:
            score += 3  # Kategorik değişkenler hedef olabilir
        
        # Entropi bazlı puanlama (düşük entropi = daha iyi hedef değişken)
        entropy_score = 1 - (info['entropy'] / np.log2(info['n_unique']))
        score += entropy_score * 2
        
        # Son sütun olma durumu
        if info['is_last_column']:
            score += 1  # Son sütun genellikle hedef değişkendir
        
        # Eksik veri oranı (düşük olması tercih edilir)
        missing_penalty = info['missing_ratio'] * 3
        score -= missing_penalty
        
        # Sayısal değişkenler için ek kontroller
        if info['is_numeric']:
            # Eğer değerler 0-1 arasındaysa
            if df[col].between(0, 1).all():
                score += 2
            # Eğer değerler tam sayıysa
            elif df[col].apply(lambda x: float(x).is_integer() if isinstance(x, (int, float)) else False).all():
                score += 2
        
        target_scores[col] = score
    
    # En yüksek puanlı sütunu hedef değişken olarak seç
    target_column = max(target_scores.items(), key=lambda x: x[1])[0]
    feature_columns = [col for col in df.columns if col != target_column]
    
    # Hedef değişken seçimini raporla
    print("\nHedef değişken tespiti:")
    print(f"Seçilen hedef değişken: {target_column}")
    print("\nSütun puanları:")
    for col, score in sorted(target_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{col}: {score:.2f}")
    
    # Seçilen hedef değişkenin özelliklerini raporla
    target_info = column_info[target_column]
    print(f"\nHedef değişken özellikleri:")
    print(f"- Benzersiz değer sayısı: {target_info['n_unique']}")
    print(f"- Binary değişken: {'Evet' if target_info['is_binary'] else 'Hayır'}")
    print(f"- Kategorik değişken: {'Evet' if target_info['is_categorical'] else 'Hayır'}")
    print(f"- Sayısal değişken: {'Evet' if target_info['is_numeric'] else 'Hayır'}")
    print(f"- Eksik veri oranı: {target_info['missing_ratio']:.2%}")
    
    return target_column, feature_columns

def evaluate_knn_model(model_file_path, test_data_path):
    """
    KNN modelini değerlendir ve doğruluk oranını hesapla
    
    Args:
        model_file_path: Model dosyasının yolu (.pkl veya .joblib)
        test_data_path: Test veri seti dosyasının yolu (.csv)
    
    Returns:
        float: Doğruluk oranı (0-1 arası)
    """
    try:
        # Model dosyasını yükle (hem pickle hem joblib formatını dene)
        model = None
        try:
            # Önce pickle ile dene
            with open(model_file_path, 'rb') as f:
                model = pickle.load(f)
        except:
            try:
                # Pickle başarısız olursa joblib ile dene
                model = joblib.load(model_file_path)
            except Exception as e:
                print(f"Model yükleme hatası: {str(e)}")
                print("Model dosyası geçerli bir scikit-learn modeli içermiyor.")
                print("Desteklenen formatlar: .pkl (pickle) veya .joblib")
                return None

        # Model tipini kontrol et
        if not isinstance(model, BaseEstimator):
            print("Yüklenen dosya geçerli bir scikit-learn modeli değil.")
            print("Model, sklearn.base.BaseEstimator sınıfından türetilmiş olmalıdır.")
            return None

        # Test verisini yükle
        try:
            test_data = pd.read_csv(test_data_path)
            print("\nTest veri seti yüklendi:")
            print(f"Sütunlar: {', '.join(test_data.columns)}")
            print(f"Örnek sayısı: {len(test_data)}")
            print(f"Veri seti boyutu: {test_data.shape}")
        except Exception as e:
            print(f"Test veri seti yükleme hatası: {str(e)}")
            print("Test veri seti geçerli bir CSV dosyası olmalıdır.")
            return None

        # Hedef değişkeni akıllı tespit et
        target_column, feature_columns = detect_target_column(test_data)
        print(f"\nÖzellik sütunları: {', '.join(feature_columns)}")

        X_test = test_data[feature_columns]
        y_test = test_data[target_column]

        # Model özellik sayısını kontrol et
        if hasattr(model, 'n_features_in_'):
            if model.n_features_in_ != X_test.shape[1]:
                print(f"\nUyarı: Model özellik sayısı uyuşmuyor!")
                print(f"Model {model.n_features_in_} özellik bekliyor, test verisi {X_test.shape[1]} özellik içeriyor.")
                print("Özellik isimleri:")
                print(f"Model beklenen özellikler: {model.feature_names_in_ if hasattr(model, 'feature_names_in_') else 'Bilinmiyor'}")
                print(f"Test verisi özellikleri: {feature_columns}")
                return None

        # Tahminleri yap ve doğruluk oranını hesapla
        try:
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"\nModel değerlendirme başarılı!")
            print(f"Test veri seti boyutu: {len(X_test)} örnek")
            print(f"Özellik sayısı: {X_test.shape[1]}")
            print(f"Doğruluk oranı: {accuracy:.2%}")
            
            # Sınıf dağılımını göster
            print("\nSınıf dağılımı:")
            print("Gerçek değerler:")
            print(pd.Series(y_test).value_counts().to_dict())
            print("\nTahmin edilen değerler:")
            print(pd.Series(y_pred).value_counts().to_dict())
            
            # Karmaşıklık matrisi (eğer sınıf sayısı azsa)
            if len(np.unique(y_test)) <= 10:
                from sklearn.metrics import confusion_matrix, classification_report
                print("\nKarmaşıklık Matrisi:")
                print(confusion_matrix(y_test, y_pred))
                print("\nSınıflandırma Raporu:")
                print(classification_report(y_test, y_pred))
            
            return accuracy
        except Exception as e:
            print(f"Model değerlendirme hatası: {str(e)}")
            print("Model ve test veri seti uyumsuz olabilir.")
            print("Lütfen modelinizin test veri setiyle aynı özellik yapısına sahip olduğundan emin olun.")
            return None

    except Exception as e:
        print(f"Beklenmeyen hata: {str(e)}")
        print("Hata detayı:")
        print(traceback.format_exc())
        return None

def update_model_evaluation(user_model):
    """
    Kullanıcı modelini değerlendir ve sonuçları güncelle
    
    Args:
        user_model: UserModel instance
    """
    try:
        if not user_model.model_file or not user_model.test_data or not user_model.test_data.file:
            print("Model veya test veri seti eksik.")
            return

        # Model ve test veri seti dosya yollarını al
        model_path = user_model.model_file.path
        test_data_path = user_model.test_data.file.path

        print(f"\nModel değerlendirmesi başlatılıyor...")
        print(f"Model dosyası: {user_model.model_file.name}")
        print(f"Test veri seti: {user_model.test_data.name}")

        # Modeli değerlendir
        accuracy = evaluate_knn_model(model_path, test_data_path)

        if accuracy is not None:
            # Sonuçları güncelle
            user_model.accuracy = accuracy
            user_model.evaluated_at = datetime.now()
            user_model.save()
            print(f"\nModel değerlendirmesi tamamlandı ve kaydedildi.")
        else:
            print("\nModel değerlendirilemedi.")
            print("Lütfen model dosyanızın ve test veri setinin doğru formatta olduğundan emin olun.")

    except Exception as e:
        print(f"Model güncelleme hatası: {str(e)}")
        print("Hata detayı:")
        print(traceback.format_exc()) 