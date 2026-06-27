from django.contrib import admin
from .models import SignHistory, Sentence


@admin.register(SignHistory)
class SignHistoryAdmin(admin.ModelAdmin):
    list_display = ['sign', 'meaning', 'category', 'confidence', 'session_id', 'detected_at']
    list_filter = ['category', 'detected_at']
    search_fields = ['sign', 'meaning', 'session_id']
    ordering = ['-detected_at']
    readonly_fields = ['detected_at']


@admin.register(Sentence)
class SentenceAdmin(admin.ModelAdmin):
    list_display = ['text', 'session_id', 'created_at']
    search_fields = ['text', 'session_id']
    ordering = ['-created_at']
    readonly_fields = ['created_at']
