from django.db import models


class SignHistory(models.Model):
    """Stores each detected sign with metadata"""
    sign = models.CharField(max_length=50)
    meaning = models.CharField(max_length=100)
    category = models.CharField(max_length=50)
    confidence = models.FloatField()
    session_id = models.CharField(max_length=100, blank=True)
    detected_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-detected_at']

    def __str__(self):
        return f"{self.sign} ({self.confidence:.0%}) @ {self.detected_at.strftime('%H:%M:%S')}"


class Sentence(models.Model):
    """Stores composed sentences from the sentence builder"""
    text = models.TextField()
    session_id = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.text[:60]
