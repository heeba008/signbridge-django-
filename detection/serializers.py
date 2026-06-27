from rest_framework import serializers
from .models import SignHistory, Sentence


class SignHistorySerializer(serializers.ModelSerializer):
    detected_at = serializers.DateTimeField(format="%Y-%m-%d %H:%M:%S", read_only=True)

    class Meta:
        model = SignHistory
        fields = ['id', 'sign', 'meaning', 'category', 'confidence', 'session_id', 'detected_at']


class SentenceSerializer(serializers.ModelSerializer):
    created_at = serializers.DateTimeField(format="%Y-%m-%d %H:%M:%S", read_only=True)

    class Meta:
        model = Sentence
        fields = ['id', 'text', 'session_id', 'created_at']
