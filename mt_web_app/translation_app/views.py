from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .services import ensemble

def index(request):
    return render(request, "translation_app/index.html")

@csrf_exempt
def translate_text(request):
    if request.method == "POST":
        text = request.POST.get("text", "").strip()
        if not text: return JsonResponse({"error": "No text provided"})
        try:
            translation, selected_model, score = ensemble.translate(text)
            return JsonResponse({"translation": translation, "model_used": selected_model, "comet_score": float(score)})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Invalid request method"}, status=400)
