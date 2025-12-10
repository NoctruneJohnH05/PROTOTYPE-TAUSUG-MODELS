from django.template import loader
from django.shortcuts import render
from django.http import HttpResponseRedirect, JsonResponse
from .utils import get_top_3_preds, get_top_3_preds_bidirectional
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