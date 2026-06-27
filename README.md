# SignBridge — Full-Stack Django + MediaPipe

A full-stack web application for real-time ASL sign language detection.
Frontend: HTML/CSS/JS + MediaPipe | Backend: Python + Django + SQLite

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install django djangorestframework

# 2. Run migrations (creates SQLite DB)
python manage.py migrate

# 3. Create admin user (optional)
python manage.py createsuperuser

# 4. Start server
python manage.py runserver

# 5. Open browser
# App:       http://127.0.0.1:8000/
# Dashboard: http://127.0.0.1:8000/dashboard/
# Admin:     http://127.0.0.1:8000/admin/
```

---

## 📁 Project Structure

```
signbridge_project/
├── manage.py                   ← Django entry point
├── db.sqlite3                  ← SQLite database (auto-created)
├── signbridge_project/
│   ├── settings.py             ← Django config
│   └── urls.py                 ← Root URL routing
└── detection/
    ├── models.py               ← SignHistory + Sentence models
    ├── views.py                ← Page views + REST API views
    ├── serializers.py          ← DRF serializers
    ├── urls.py                 ← App URL routing
    ├── admin.py                ← Django admin config
    ├── templates/detection/
    │   ├── index.html          ← Main app (Django template)
    │   └── dashboard.html      ← Analytics dashboard
    └── static/detection/
        ├── css/style.css       ← Styling
        └── js/
            ├── app.js          ← MediaPipe + gesture logic
            └── django_api.js   ← Django REST API integration
```

---

## 🔌 REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/history/` | List all detected signs |
| POST | `/api/history/` | Save a detected sign |
| DELETE | `/api/history/clear/` | Clear all history |
| GET | `/api/stats/` | Aggregate stats |
| GET | `/api/sentences/` | List saved sentences |
| POST | `/api/sentences/` | Save a sentence |

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Django 4.x |
| REST API | Django REST Framework |
| Database | SQLite (via Django ORM) |
| Frontend | HTML5, CSS3, JavaScript |
| CV/ML | MediaPipe Hands |
| Speech | Web Speech API |
| Camera | WebRTC / getUserMedia |

---

## 📋 Resume Skills This Covers

- Python · Django · Django REST Framework
- SQLite · ORM · Database Migrations
- REST API Design (GET/POST/DELETE)
- Full-Stack Development
- MediaPipe · Computer Vision
- JavaScript · HTML · CSS
