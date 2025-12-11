from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import psutil
import os
from .utils import (
    get_top_3_preds,
    get_top_3_preds_bidirectional,
    get_top_3_preds_gru,
    generate_text_lstm,
    generate_text_bidirectional,
    generate_text_gru
)


# -------------------------------
# Helper function: memory check
# -------------------------------
def check_memory_available():
    """Check if enough memory is available"""
    try:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        if memory_mb > 450:  # warning threshold
            print(f"⚠️ [MEMORY WARNING] Using {memory_mb:.0f}MB / 512MB")
            return False
        return True
    except:
        return True  # If psutil fails, continue anyway


# -------------------------------
# Root page / document editor
# -------------------------------
def render_doc_editor(request):
    if request.method == "POST":
        data = json.loads(request.body.decode("utf-8"))
        input_text = data.get("data", "")
        model_type = data.get("model", "lstm")

        if model_type == "bidirectional":
            top_3_preds = get_top_3_preds_bidirectional(
                prompt=input_text,
                model_path="autocomplete/BIDIRECTIONAL-FINETUNED2.keras"
            )
        elif model_type == "gru":
            top_3_preds = get_top_3_preds_gru(
                prompt=input_text,
                model_path="autocomplete/GRU.pt"
            )
        else:
            top_3_preds = get_top_3_preds(
                prompt=input_text,
                model_path="autocomplete/LSTM-TESTING.keras"
            )

        return JsonResponse({"preds": top_3_preds, "model_used": model_type})

    return render(request, "autocomplete/interface.html")


# -------------------------------
# Predict next word API
# -------------------------------
@csrf_exempt
def predict_next_word(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST method required"}, status=405)

    try:
        if not check_memory_available():
            return JsonResponse(
                {"error": "Server memory low", "preds": "", "model_used": "none"},
                status=503
            )

        data = json.loads(request.body)
        input_text = data.get("data", "")
        model_choice = data.get("model", "lstm")

        if not input_text:
            return JsonResponse(
                {"error": "No input text provided", "preds": "", "model_used": "none"},
                status=400
            )

        # Select model
        if model_choice == "bidirectional":
            top_3_preds = get_top_3_preds_bidirectional(
                prompt=input_text,
                model_path="autocomplete/BIDIRECTIONAL-FINETUNED2.keras"
            )
        elif model_choice == "gru":
            top_3_preds = get_top_3_preds_gru(
                prompt=input_text,
                model_path="autocomplete/GRU.pt"
            )
        else:
            top_3_preds = get_top_3_preds(
                prompt=input_text,
                model_path="autocomplete/LSTM-TESTING.keras"
            )

        return JsonResponse({"preds": top_3_preds, "model_used": model_choice})

    except MemoryError:
        return JsonResponse(
            {"error": "Server out of memory", "preds": "", "model_used": "none"},
            status=503
        )
    except Exception as e:
        return JsonResponse(
            {"error": str(e), "preds": "", "model_used": "none"}, status=500
        )


# -------------------------------
# Text generation API
# -------------------------------
@csrf_exempt
def generate_text(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST method required"}, status=405)

    try:
        if not check_memory_available():
            return JsonResponse(
                {"error": "Server memory low", "generated_text": ""},
                status=503
            )

        data = json.loads(request.body)
        input_text = data.get("data", "")
        model_choice = data.get("model", "lstm")
        num_words = int(data.get("num_words", 20))
        temperature = float(data.get("temperature", 0.7))

        if not input_text:
            return JsonResponse(
                {"error": "No input text provided", "generated_text": ""},
                status=400
            )

        # Select generation model
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
        else:
            generated_text = generate_text_lstm(
                prompt=input_text,
                model_path="autocomplete/LSTM-TESTING.keras",
                num_words=num_words,
                temperature=temperature
            )

        return JsonResponse({"generated_text": generated_text, "model_used": model_choice})

    except MemoryError:
        return JsonResponse(
            {"error": "Server out of memory", "generated_text": ""},
            status=503
        )
    except Exception as e:
        return JsonResponse({"error": str(e), "generated_text": ""}, status=500)


# -------------------------------
# Placeholder Autocomplete API
# -------------------------------
@csrf_exempt
def autocomplete(request):
    return JsonResponse({"status": "ok", "message": "Autocomplete endpoint working"})


# -------------------------------
# Placeholder Translate API
# -------------------------------
@csrf_exempt
def translate(request):
    return JsonResponse({"status": "ok", "message": "Translate endpoint working"})
