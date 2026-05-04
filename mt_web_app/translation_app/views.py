from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .services import ensemble

def index(request):
    return render(request, "translation_app/index.html")

STATIC_METRICS = {
    "nllb": {"bleu": 39.92, "meteor": 0.6488, "bertscore": 0.9000, "comet": 0.8158},
    "indic": {"bleu": 29.24, "meteor": 0.5389, "bertscore": 0.8425, "comet": 0.6426},
    "ensemble": {"bleu": 39.89, "meteor": 0.6480, "bertscore": 0.8991, "comet": 0.8148}
}

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
                "comet_score": score_display,
                "static_metrics": STATIC_METRICS.get(model_choice, STATIC_METRICS["ensemble"])
            }
            if metrics_dict:
                response_data["metrics"] = metrics_dict
                
            return JsonResponse(response_data)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Invalid request method"}, status=400)
