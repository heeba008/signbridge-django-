from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import SignHistory, Sentence
from .serializers import SignHistorySerializer, SentenceSerializer
from .predictor import get_predictor
import json


# ── Frontend View ──────────────────────────────────────────────────────────────

def index(request):
    """Serve the main SignBridge app"""
    return render(request, 'detection/index.html')


def dashboard(request):
    """Stats dashboard — shows all saved history from the DB"""
    total_signs = SignHistory.objects.count()
    top_signs = (
        SignHistory.objects
        .values('sign', 'meaning', 'category')
        .annotate(count=__import__('django.db.models', fromlist=['Count']).Count('id'))
        .order_by('-count')[:10]
    )
    recent = SignHistory.objects.all()[:20]
    sentences = Sentence.objects.all()[:10]

    return render(request, 'detection/dashboard.html', {
        'total_signs': total_signs,
        'top_signs': top_signs,
        'recent': recent,
        'sentences': sentences,
    })


# ── REST API: ML Prediction ───────────────────────────────────────────────────

@api_view(['POST'])
def predict(request):
    """
    POST /api/predict/

    Accepts 21 MediaPipe hand landmarks and returns the predicted ASL sign
    with confidence score from the trained scikit-learn model.

    Request body:
        {
          "landmarks": [
            {"x": 0.51, "y": 0.72, "z": -0.03},
            ...   (21 points total)
          ]
        }

    Response:
        {
          "sign":        "A",
          "confidence":  0.94,
          "meaning":     "Letter A",
          "category":    "Alphabet",
          "model_used":  "RandomForest"
        }
    """
    landmarks = request.data.get("landmarks")

    if not landmarks:
        return Response({"error": "Missing 'landmarks' key"}, status=status.HTTP_400_BAD_REQUEST)

    if len(landmarks) != 21:
        return Response(
            {"error": f"Expected 21 landmarks, got {len(landmarks)}"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        predictor = get_predictor()
        sign, confidence = predictor.predict(landmarks)
        sign_info = predictor.sign_db.get(sign, {})

        return Response({
            "sign":       sign,
            "confidence": round(float(confidence), 4),
            "meaning":    sign_info.get("meaning", sign),
            "category":   sign_info.get("category", "Unknown"),
            "model_used": predictor.model_name,
        })

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def model_info(request):
    """
    GET /api/model/info/

    Returns metadata about the currently loaded ML model.
    Useful for debugging and showing model status in the dashboard.
    """
    predictor = get_predictor()
    return Response({
        "model_name":    predictor.model_name,
        "model_loaded":  predictor._use_model,
        "test_accuracy": predictor.accuracy,
        "model_path":    predictor.model_path,
        "num_classes":   len(predictor.sign_db),
    })


# ── REST API: Sign History ─────────────────────────────────────────────────────

@api_view(['GET', 'POST'])
def sign_history(request):
    """
    GET  /api/history/   → list all saved signs
    POST /api/history/   → save a new detected sign
    """
    if request.method == 'GET':
        session_id = request.query_params.get('session_id', None)
        qs = SignHistory.objects.all()
        if session_id:
            qs = qs.filter(session_id=session_id)
        serializer = SignHistorySerializer(qs[:100], many=True)
        return Response({'count': qs.count(), 'results': serializer.data})

    elif request.method == 'POST':
        serializer = SignHistorySerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['DELETE'])
def clear_history(request):
    """DELETE /api/history/clear/  → wipe all sign history"""
    session_id = request.query_params.get('session_id', None)
    if session_id:
        deleted, _ = SignHistory.objects.filter(session_id=session_id).delete()
    else:
        deleted, _ = SignHistory.objects.all().delete()
    return Response({'deleted': deleted})


# ── REST API: Stats ────────────────────────────────────────────────────────────

@api_view(['GET'])
def stats(request):
    """GET /api/stats/  → aggregate stats across all sessions"""
    from django.db.models import Avg, Count

    data = {
        'total_signs': SignHistory.objects.count(),
        'avg_confidence': SignHistory.objects.aggregate(avg=Avg('confidence'))['avg'] or 0,
        'total_sentences': Sentence.objects.count(),
        'by_category': list(
            SignHistory.objects
            .values('category')
            .annotate(count=Count('id'))
            .order_by('-count')
        ),
        'top_signs': list(
            SignHistory.objects
            .values('sign', 'meaning')
            .annotate(count=Count('id'))
            .order_by('-count')[:5]
        ),
    }
    data['avg_confidence'] = round(data['avg_confidence'] * 100, 1)
    return Response(data)


# ── REST API: Sentences ────────────────────────────────────────────────────────

@api_view(['GET', 'POST'])
def sentences(request):
    """
    GET  /api/sentences/  → list saved sentences
    POST /api/sentences/  → save a new sentence
    """
    if request.method == 'GET':
        qs = Sentence.objects.all()[:50]
        serializer = SentenceSerializer(qs, many=True)
        return Response({'results': serializer.data})

    elif request.method == 'POST':
        serializer = SentenceSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
