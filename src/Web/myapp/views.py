from django.conf import settings
from django.shortcuts import render, redirect
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
            model_file = request.FILES.get('model_file')
            test_data_id = request.POST.get('test_data')
            
            # Dosya uzantısını kontrol et
            if model_file:
                file_ext = os.path.splitext(model_file.name)[1].lower()
                if file_ext not in ['.pkl', '.joblib']:
                    context['message'] = (
                        "<strong>Hata: Geçersiz Dosya Formatı!</strong><br><br>"
                        "Lütfen .pkl veya .joblib uzantılı bir model dosyası yükleyin.<br>"
                        "Desteklenen formatlar:<br>"
                        "- .pkl (Python pickle formatı)<br>"
                        "- .joblib (scikit-learn joblib formatı)"
                    )
                    context['success'] = False
                    context['test_data'] = TestData.objects.filter(is_active=True)
                    return render(request, 'myapp/upload_model.html', context)
            
            if model_file and test_data_id:
                try:
                    # Test veri setini kontrol et
                    test_data = TestData.objects.get(id=test_data_id, is_active=True)
                    
                    # Model kaydını oluştur
                    user_model = UserModel.objects.create(
                        email=email,
                        school_number=school_number,
                        first_name=first_name,
                        last_name=last_name,
                        model_file=model_file,
                        test_data=test_data
                    )
                    
                    # Modeli değerlendir
                    update_model_evaluation(user_model)
                    
                    if user_model.accuracy is not None:
                        context['message'] = (
                            f"<strong>Model Başarıyla Yüklendi!</strong><br><br>"
                            f"<strong>Model Adı:</strong> {model_file.name}<br>"
                            f"<strong>Test Veri Seti:</strong> {test_data.name}<br>"
                            f"<strong>Doğruluk Oranı:</strong> {user_model.accuracy:.2%}<br><br>"
                            f"Sonuçları görüntülemek için 'Model Sonuçları' sayfasını kullanabilirsiniz."
                        )
                        context['success'] = True
                    else:
                        context['message'] = (
                            "<strong>Model Yüklendi Ancak Değerlendirilemedi!</strong><br><br>"
                            "Model dosyası yüklendi ancak test veri seti üzerinde değerlendirilemedi.<br><br>"
                            "<strong>Olası nedenler:</strong><br>"
                            "1. Model dosyası geçerli bir scikit-learn modeli içermiyor olabilir<br>"
                            "2. Model ve test veri seti uyumsuz olabilir (özellik sayıları farklı)<br>"
                            "3. Test veri seti formatı uygun olmayabilir<br><br>"
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
                context['message'] = "Lütfen tüm alanları doldurun ve bir model dosyası seçin."
                context['success'] = False
        
        except Exception as e:
            context['message'] = f"Beklenmeyen bir hata oluştu: {str(e)}"
            context['success'] = False
            print("Hata detayı:")
            print(traceback.format_exc())
    
    # Aktif test veri setlerini getir
    context['test_data'] = TestData.objects.filter(is_active=True)
    
    return render(request, 'myapp/upload_model.html', context)


