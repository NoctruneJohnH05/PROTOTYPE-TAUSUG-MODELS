from django.template import loader
from django.shortcuts import render
from django.http import HttpResponseRedirect, JsonResponse
from .utils import (get_top_3_preds, get_top_3_preds_bidirectional, get_top_3_preds_gru,
                    generate_text_lstm, generate_text_bidirectional, generate_text_gru)
import json


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


def generate_text(request):
    """
    Generate continuous text based on a prompt.
    Accepts POST requests with: data (prompt), model (lstm/bidirectional/gru), 
    num_words (length), and temperature (randomness).
    """
    if request.method == "POST":
        try:
            raw_body = request.body.decode('utf-8')
            data = json.loads(raw_body)
            
            input_text = data.get("data", "")
            model_type = data.get("model", "lstm")
            num_words = int(data.get("num_words", 20))
            temperature = float(data.get("temperature", 1.0))
            
            print(f"\n{'='*60}")
            print(f"GENERATION REQUEST")
            print(f"MODEL: {model_type.upper()}")
            print(f"Prompt: {input_text}")
            print(f"Words to generate: {num_words}")
            print(f"Temperature: {temperature}")
            print(f"{'='*60}")
            
            # Validate parameters
            num_words = max(1, min(num_words, 100))  # Limit between 1-100 words
            temperature = max(0.1, min(temperature, 2.0))  # Limit between 0.1-2.0
            
            # Generate text based on selected model
            if model_type == "bidirectional":
                generated_text = generate_text_bidirectional(
                    prompt=input_text,
                    model_path="autocomplete/BIDIRECTIONAL-FINETUNED2.keras",
                    num_words=num_words,
                    temperature=temperature
                )
            elif model_type == "gru":
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
            
            print(f"Generated: {generated_text}")
            print(f"{'='*60}\n")
            
            return JsonResponse({
                "generated_text": generated_text,
                "model_used": model_type,
                "num_words": num_words,
                "temperature": temperature
            })
        
        except Exception as e:
            print(f"Error in text generation: {e}")
            return JsonResponse({"error": str(e)}, status=500)
    
    return JsonResponse({"error": "Only POST requests allowed"}, status=405)
