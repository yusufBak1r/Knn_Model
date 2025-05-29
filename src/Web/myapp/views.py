from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse
import os

def uploadView(request):
    context = {}
    
    if request.method == 'POST':
        if request.FILES.get('file'):
            uploaded_file = request.FILES['file']
            email = request.POST.get('email')
            
            # Define the directory where uploaded files will be stored
            file_path = os.path.join(settings.DOWNLOAD_DIR, uploaded_file.name)

            # Make sure the directory exists
            os.makedirs(settings.DOWNLOAD_DIR, exist_ok=True)

            # Save the uploaded file
            with open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            # Prepare a success message
            context['message'] = f"Dosya '{uploaded_file.name}' başarıyla yüklendi. E-posta: {email}"
        else:
            context['message'] = "Lütfen bir dosya seçin."
    
    # Return the upload page with any context (success/error message)
    return render(request, 'upload.html', context)

# def homeView(request):
#     return render(request,"home.html")

# def aboutView(request):
#     return render(request,"about.html")

