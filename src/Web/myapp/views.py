from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse
from django.contrib import messages
import os
from .models import UserModel, TestData
from django.views.decorators.csrf import csrf_exempt
from .utils import update_model_evaluation
import pickle
import traceback

def modelResultsView(request):
    context = {}
    
    if request.method == 'POST':
        # Form verilerini al
        email = request.POST.get('email')
        school_number = request.POST.get('schoolNumber')
        
        # Kullanıcının modellerini bul
        user_models = UserModel.objects.filter(
            email=email,
            school_number=school_number
        ).order_by('-uploaded_at')
        
        if user_models.exists():
            context['models'] = user_models
            context['has_results'] = True
        else:
            context['message'] = "Bu bilgilerle kayıtlı model bulunamadı."
            context['success'] = False
    
    # Aktif test veri setlerini getir
    context['test_data'] = TestData.objects.filter(is_active=True)
    
    return render(request, 'myapp/model_results.html', context)

@csrf_exempt
def uploadModelView(request):
    context = {}
    
    if request.method == 'POST':
        try:
            # Form verilerini al
            email = request.POST.get('email')
            school_number = request.POST.get('school_number')
            first_name = request.POST.get('first_name')
            last_name = request.POST.get('last_name')
            algorithm_type = request.POST.get('algorithm_type')
            model_file = request.FILES.get('model_file')
            test_data_id = request.POST.get('test_data')
            
            # Algoritma tipi kontrolü
            if not algorithm_type or algorithm_type not in ['knn', 'naive_bayes']:
                context['message'] = (
                    "<strong>Hata: Algoritma Seçimi Gerekli!</strong><br><br>"
                    "Lütfen KNN veya Naive Bayes algoritmasından birini seçin."
                )
                context['success'] = False
                context['test_data'] = TestData.objects.filter(is_active=True)
                return render(request, 'myapp/upload_model.html', context)
            
            # Dosya uzantısını kontrol et
            if model_file:
                file_ext = os.path.splitext(model_file.name)[1].lower()
                if file_ext not in ['.pkl', '.joblib']:
                    algorithm_display = "KNN" if algorithm_type == 'knn' else "Naive Bayes"
                    context['message'] = (
                        "<strong>Hata: Geçersiz Dosya Formatı!</strong><br><br>"
                        f"Lütfen .pkl veya .joblib uzantılı bir {algorithm_display} model dosyası yükleyin.<br>"
                        "Desteklenen formatlar:<br>"
                        "- .pkl (Python pickle formatı)<br>"
                        "- .joblib (scikit-learn joblib formatı)"
                    )
                    context['success'] = False
                    context['test_data'] = TestData.objects.filter(is_active=True)
                    return render(request, 'myapp/upload_model.html', context)
            
            if model_file and test_data_id and algorithm_type:
                try:
                    # Test veri setini kontrol et
                    test_data = TestData.objects.get(id=test_data_id, is_active=True)
                    
                    # Model kaydını oluştur
                    user_model = UserModel.objects.create(
                        email=email,
                        school_number=school_number,
                        first_name=first_name,
                        last_name=last_name,
                        algorithm_type=algorithm_type,
                        model_file=model_file,
                        test_data=test_data
                    )
                    
                    # Modeli değerlendir
                    update_model_evaluation(user_model)
                    
                    # Algoritma tipini kullanıcı dostu formatta göster
                    algorithm_display = user_model.get_algorithm_type_display()
                    
                    if user_model.accuracy is not None:
                        context['message'] = (
                            f"<strong>Model Başarıyla Yüklendi!</strong><br><br>"
                            f"<strong>Algoritma:</strong> {algorithm_display}<br>"
                            f"<strong>Model Adı:</strong> {model_file.name}<br>"
                            f"<strong>Test Veri Seti:</strong> {test_data.name}<br>"
                            f"<strong>Doğruluk Oranı:</strong> {user_model.accuracy:.2%}<br><br>"
                            f"Sonuçları görüntülemek için 'Model Sonuçları' sayfasını kullanabilirsiniz."
                        )
                        context['success'] = True
                    else:
                        context['message'] = (
                            "<strong>Model Yüklendi Ancak Değerlendirilemedi!</strong><br><br>"
                            f"<strong>Algoritma:</strong> {algorithm_display}<br>"
                            f"Model dosyası yüklendi ancak test veri seti üzerinde değerlendirilemedi.<br><br>"
                            "<strong>Olası nedenler:</strong><br>"
                            f"1. Model dosyası geçerli bir {algorithm_display} modeli içermiyor olabilir<br>"
                            "2. Model ve test veri seti uyumsuz olabilir (özellik sayıları farklı)<br>"
                            "3. Test veri seti formatı uygun olmayabilir<br>"
                            "4. Naive Bayes için veri tipi uyumsuzluğu olabilir<br><br>"
                            "Lütfen modelinizin ve test veri setinin doğru formatta olduğundan emin olun."
                        )
                        context['success'] = False
                    
                except Exception as e:
                    context['message'] = (
                        f"<strong>Model Yükleme Hatası!</strong><br><br>"
                        f"Hata detayı: {str(e)}<br><br>"
                        "Lütfen tekrar deneyin veya sistem yöneticisi ile iletişime geçin."
                    )
                    context['success'] = False
                    print("Hata detayı:")
                    print(traceback.format_exc())
            else:
                context['message'] = "Lütfen tüm alanları doldurun, bir algoritma seçin ve bir model dosyası seçin."
                context['success'] = False
        
        except Exception as e:
            context['message'] = f"Beklenmeyen bir hata oluştu: {str(e)}"
            context['success'] = False
            print("Hata detayı:")
            print(traceback.format_exc())
    
    # Aktif test veri setlerini getir
    context['test_data'] = TestData.objects.filter(is_active=True)
    
    return render(request, 'myapp/upload_model.html', context)

def confusionMatrixView(request, model_id):
    """
    Belirli bir modelin confusion matrix'ini görüntüle
    """
    context = {}
    
    try:
        # Model'i al
        user_model = get_object_or_404(UserModel, id=model_id)
        
        # Confusion matrix verilerini kontrol et
        if not user_model.confusion_matrix or not user_model.class_labels:
            # Eğer confusion matrix verisi yoksa, modeli yeniden değerlendir
            try:
                print(f"Model {user_model.id} için confusion matrix verisi bulunamadı. Yeniden değerlendiriliyor...")
                update_model_evaluation(user_model)
                user_model.refresh_from_db()  # Veritabanından güncel veriyi al
                
                if not user_model.confusion_matrix or not user_model.class_labels:
                    context['error'] = "Model yeniden değerlendirildi ancak confusion matrix verisi oluşturulamadı. Model veya test veri seti ile ilgili bir sorun olabilir."
                    context['model'] = user_model
                    return render(request, 'myapp/confusion_matrix.html', context)
            except Exception as e:
                context['error'] = f"Model yeniden değerlendirilirken hata oluştu: {str(e)}"
                context['model'] = user_model
                return render(request, 'myapp/confusion_matrix.html', context)
        
        # Verileri template'e gönder
        context['model'] = user_model
        context['confusion_matrix'] = user_model.confusion_matrix
        context['class_labels'] = user_model.class_labels
        context['classification_report'] = user_model.classification_report
        context['success'] = True
        
        # Confusion matrix boyutunu hesapla
        matrix_size = len(user_model.class_labels)
        context['matrix_size'] = matrix_size
        
        # Toplam örnekleri hesapla
        total_samples = sum(sum(row) for row in user_model.confusion_matrix)
        context['total_samples'] = total_samples
        
        # Confusion Matrix doğrulama ve ek bilgi hesaplama
        correct_predictions = sum(user_model.confusion_matrix[i][i] for i in range(len(user_model.confusion_matrix)))
        incorrect_predictions = total_samples - correct_predictions
        
        context['correct_predictions'] = correct_predictions
        context['incorrect_predictions'] = incorrect_predictions
        context['manual_accuracy'] = (correct_predictions / total_samples) if total_samples > 0 else 0
        
        # Manuel accuracy ile model accuracy karşılaştırma (doğrulama için)
        accuracy_diff = abs(context['manual_accuracy'] - user_model.accuracy) if user_model.accuracy else 0
        context['accuracy_verification'] = {
            'manual_accuracy': context['manual_accuracy'],
            'model_accuracy': user_model.accuracy,
            'difference': accuracy_diff,
            'is_valid': accuracy_diff < 0.001  # Tolerans: 0.1%
        }
        
        # Her hücre için yüzde hesapla
        confusion_matrix_with_percentages = []
        for i, row in enumerate(user_model.confusion_matrix):
            row_with_percentages = []
            for j, value in enumerate(row):
                percentage = (value / total_samples * 100) if total_samples > 0 else 0
                row_with_percentages.append({
                    'value': value,
                    'percentage': percentage,
                    'is_diagonal': i == j  # Doğru tahminler için
                })
            
            # Bu sınıf için performans bilgilerini bul
            class_performance_data = None
            if user_model.classification_report and user_model.confusion_matrix:
                if i < len(user_model.class_labels):
                    class_label = user_model.class_labels[i]
                    class_label_str = str(class_label)
                    if class_label_str in user_model.classification_report:
                        metrics = user_model.classification_report[class_label_str]
                        
                        # Bu sınıf için True Positive, False Positive, False Negative hesapla
                        tp = user_model.confusion_matrix[i][i]  # Doğru tahminler (köşegen elemanlar)
                        
                        # False Positive: Bu sınıf olarak yanlış tahmin edilenler (sütun toplamı - TP)
                        fp = sum(user_model.confusion_matrix[k][i] for k in range(len(user_model.confusion_matrix)) if k != i)
                        
                        # False Negative: Bu sınıftan diğer sınıflara yanlış tahmin edilenler (satır toplamı - TP)
                        fn = sum(user_model.confusion_matrix[i][k] for k in range(len(user_model.confusion_matrix[i])) if k != i)
                        
                        # True Negative: Diğer sınıfların doğru tahmin edilme sayısı
                        tn = total_samples - tp - fp - fn
                        
                        class_performance_data = {
                            'tp': tp,
                            'fp': fp,
                            'fn': fn,
                            'tn': tn,
                            'precision': metrics.get('precision', 0),
                            'recall': metrics.get('recall', 0),
                            'f1_score': metrics.get('f1_score', 0),
                            'specificity': metrics.get('specificity', 0),
                            'support': metrics.get('support', 0)
                        }
            
            confusion_matrix_with_percentages.append({
                'class_label': user_model.class_labels[i] if i < len(user_model.class_labels) else str(i),
                'cells': row_with_percentages,
                'performance': class_performance_data
            })
        
        context['confusion_matrix_with_percentages'] = confusion_matrix_with_percentages
        

        
    except Exception as e:
        context['error'] = f"Hata oluştu: {str(e)}"
        print("Confusion matrix görüntüleme hatası:")
        print(traceback.format_exc())
    
    return render(request, 'myapp/confusion_matrix.html', context)


