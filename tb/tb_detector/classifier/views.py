from django.shortcuts import render
from .model_loader import predict_fileobj

def home(request):
    context = {"prediction": None, "confidence": None}
    if request.method == "POST" and request.FILES.get("image"):
        fileobj = request.FILES["image"]
        result = predict_fileobj(fileobj, threshold=0.70)  # adjust threshold if needed
        context["prediction"] = result["label"]
        context["confidence"] = result["confidence"] * 100.0  # percentage
        # Optional: also pass raw probs for debugging
        context["probs"] = result["probs"]
    return render(request, "home.html", context)
