from django.template import loader
from django.shortcuts import render
from django.http import HttpResponseRedirect, JsonResponse
from .utils import (get_top_3_preds, get_top_3_preds_bidirectional, get_top_3_preds_gru,
                    generate_text_lstm, generate_text_bidirectional, generate_text_gru)
import json
import psutil
import os
from django.views.decorators.csrf import csrf_exempt


def render_doc_editor(request):
    if request.method == "POST":
        raw_body = request.body.decode('utf-8')
        print(f"\n{'='*60}")
        print(f"RAW REQUEST BODY: {raw_body}")
        print(f"{'='*60}")
        
        data = json.loads(raw_body)
        input_text = data["data"]
        model_type = data.get("model", "lstm")  # Default to LSTM
        
        print(f"\n{'='*60}")
        print(f"PARSED DATA: {data}")
        print(f"MODEL SELECTED: {model_type.upper()}")
        print(f"Input Text: {input_text}")
        print(f"{'='*60}")
        
        # Use selected model
        if model_type == "bidirectional":
            print("Loading BIDIRECTIONAL-FINETUNED2.keras...")
            top_3_preds = get_top_3_preds_bidirectional(
                prompt=input_text,
                model_path="autocomplete/BIDIRECTIONAL-FINETUNED2.keras"
            )
            print(f"Bidirectional Model Predictions: {top_3_preds}")
        elif model_type == "gru":
            print("Loading GRU.pt (PyTorch)...")
            top_3_preds = get_top_3_preds_gru(
                prompt=input_text,
                model_path="autocomplete/GRU.pt"
            )
            print(f"GRU Model Predictions: {top_3_preds}")
        else:  # LSTM
            print("Loading LSTM-TESTING.keras...")
            top_3_preds = get_top_3_preds(
                prompt=input_text,
                model_path="autocomplete/LSTM-TESTING.keras"
            )
            print(f"LSTM Model Predictions: {top_3_preds}")
        
        print(f"{'='*60}\n")
        return JsonResponse({"preds": top_3_preds, "model_used": model_type})

    return render(request, 'autocomplete/interface.html')


def check_memory_available():
    """Check if enough memory is available"""
    try:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > 450:
            print(f"⚠️ [MEMORY WARNING] Using {memory_mb:.0f}MB / 512MB")
            return False
        
        return True
    except:
        return True  # If psutil fails, continue anyway

@csrf_exempt
def predict_next_word(request):
    if request.method == 'POST':
        try:
            # Memory check
            if not check_memory_available():
                return JsonResponse({
                    'error': 'Server memory low, please try again',
                    'preds': '',
                    'model_used': 'none'
                }, status=503)
            
            request_data = json.loads(request.body)
            input_text = request_data.get('data', '')
            model_choice = request_data.get('model', 'lstm')
            
            if not input_text:
                return JsonResponse({
                    'error': 'No input text provided',
                    'preds': '',
                    'model_used': 'none'
                }, status=400)
            
            print(f"[REQUEST] Text: '{input_text}' | Model: {model_choice}")
            
            # Call prediction function
            if model_choice == "bidirectional":
                print("Loading BIDIRECTIONAL-FINETUNED2.keras...")
                top_3_preds = get_top_3_preds_bidirectional(
                    prompt=input_text,
                    model_path="autocomplete/BIDIRECTIONAL-FINETUNED2.keras"
                )
                print(f"Bidirectional Model Predictions: {top_3_preds}")
            elif model_choice == "gru":
                print("Loading GRU.pt (PyTorch)...")
                top_3_preds = get_top_3_preds_gru(
                    prompt=input_text,
                    model_path="autocomplete/GRU.pt"
                )
                print(f"GRU Model Predictions: {top_3_preds}")
            else:  # LSTM
                print("Loading LSTM-TESTING.keras...")
                top_3_preds = get_top_3_preds(
                    prompt=input_text,
                    model_path="autocomplete/LSTM-TESTING.keras"
                )
                print(f"LSTM Model Predictions: {top_3_preds}")
            
            return JsonResponse({
                'preds': top_3_preds,
                'model_used': model_choice
            })
            
        except MemoryError:
            print("❌ [MEMORY ERROR] Out of memory")
            return JsonResponse({
                'error': 'Server out of memory',
                'preds': '',
                'model_used': 'none'
            }, status=503)
        
        except Exception as e:
            print(f"❌ [ERROR] {str(e)}")
            return JsonResponse({
                'error': str(e),
                'preds': '',
                'model_used': 'none'
            }, status=500)
    
    return JsonResponse({'error': 'POST method required'}, status=405)

@csrf_exempt
def generate_text(request):
    if request.method == 'POST':
        try:
            # Memory check
            if not check_memory_available():
                return JsonResponse({
                    'error': 'Server memory low, please try again',
                    'generated_text': ''
                }, status=503)
            
            request_data = json.loads(request.body)
            input_text = request_data.get('data', '')
            model_choice = request_data.get('model', 'lstm')
            num_words = int(request_data.get('num_words', 20))
            temperature = float(request_data.get('temperature', 0.7))
            
            if not input_text:
                return JsonResponse({
                    'error': 'No input text provided',
                    'generated_text': ''
                }, status=400)
            
            print(f"[GENERATION] Text: '{input_text}' | Model: {model_choice} | Words: {num_words}")
            
            # Call generation function
            if model_choice == "bidirectional":
                generated_text = generate_text_bidirectional(
                    prompt=input_text,
                    model_path="autocomplete/BIDIRECTIONAL-FINETUNED2.keras",
                    num_words=num_words,
                    temperature=temperature
                )
            elif model_choice == "gru":
                generated_text = generate_text_gru(
                    prompt=input_text,
                    model_path="autocomplete/GRU.pt",
                    num_words=num_words,
                    temperature=temperature
                )
            else:  # LSTM
                generated_text = generate_text_lstm(
                    prompt=input_text,
                    model_path="autocomplete/LSTM-TESTING.keras",
                    num_words=num_words,
                    temperature=temperature
                )
            
            return JsonResponse({
                'generated_text': generated_text,
                'model_used': model_choice
            })
            
        except MemoryError:
            print("❌ [MEMORY ERROR] Out of memory")
            return JsonResponse({
                'error': 'Server out of memory',
                'generated_text': ''
            }, status=503)
        
        except Exception as e:
            print(f"❌ [ERROR] {str(e)}")
            return JsonResponse({
                'error': str(e),
                'generated_text': ''
            }, status=500)
    
    return JsonResponse({'error': 'POST method required'}, status=405)
