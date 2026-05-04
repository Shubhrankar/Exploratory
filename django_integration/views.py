import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .services import translator

def index(request):
    """Renders the main translation UI"""
    context = {
        'bleu_score': 38.36,
        'meteor_score': 0.6043,
        'bert_score': 0.9123,
        'comet_score': 0.8848
    }
    return render(request, 'mt_app/index.html', context)

@csrf_exempt
def translate_api(request):
    """API endpoint to handle AJAX translation requests"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            source_text = data.get('text', '')
            model_choice = data.get('model', 'nllb')
            reference_text = data.get('reference', '').strip()
            
            if not source_text:
                return JsonResponse({'error': 'No text provided'}, status=400)
                
            translation = translator.translate(source_text, model_name=model_choice)
            
            response_data = {
                'translation': translation,
                'model_used': model_choice
            }
            
            if reference_text:
                try:
                    scores = translator.get_metrics(translation, reference_text, source_text)
                    response_data['scores'] = scores
                except Exception as eval_err:
                    print(f"Evaluation error: {eval_err}")
            
            return JsonResponse(response_data)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
            
    return JsonResponse({'error': 'Invalid request method'}, status=405)
