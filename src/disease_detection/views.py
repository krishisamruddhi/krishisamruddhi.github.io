from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .predictor import predict_disease

def home(request):
    return render(request, 'disease_detection/home.html')

def detect(request):
    context = {}
    if request.method == "POST" and request.FILES.get('image_upload'):
        image_upload = request.FILES['image_upload']

        # Save the uploaded file to a temporary location
        fs = FileSystemStorage()
        filepath = fs.save(image_upload.name, image_upload)
        image_url = fs.url(filepath)

        # Perform prediction
        predict_class, confidence = predict_disease(fs.path(filepath))

        if predict_class:
            context = {
                'predicted_class': predict_class,
                'confidence': f"{confidence:.2%}",
                'image_url': image_url
            }
        
    return render(request, 'disease_detection/detect.html', context)