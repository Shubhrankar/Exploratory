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
        reference = request.POST.get("reference", "").strip()
        model_choice = request.POST.get("modelChoice", "ensemble").strip()
        if not text: return JsonResponse({"error": "No text provided"})
        try:
            translation, selected_model, score, metrics_dict = ensemble.translate(text, model_choice, reference)
            score_display = "N/A (Skipped QE)" if score == 0.0 else float(score)
            
            response_data = {
                "translation": translation, 
                "model_used": selected_model, 
                "comet_score": score_display
            }
            if metrics_dict:
                response_data["metrics"] = metrics_dict
                
            return JsonResponse(response_data)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Invalid request method"}, status=400)
