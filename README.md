# Ebook to Audiobook Converter (ebook-sak)

A Docker-based audiobook generation system that converts eBooks to audiobooks with AI-powered text cleaning and Kokoro TTS.

## Features
- Upload EPUB/TXT files
- AI-powered text cleaning with Google Gemini
- Multiple file upload support
- GPU-accelerated TTS with Kokoro
- Video generation with chapter markers
- Real-time progress tracking

## Prerequisites
- Docker and Docker Compose
- NVIDIA GPU with drivers (optional, for GPU acceleration)
- Google Gemini API key

## Quick Start

1. Clone the repository
2. Copy `.env.example` to `.env` and add your API key
3. Run `docker-compose up --build`
4. Access the web interface at http://localhost:5000

## For Developers
- Flask app runs in debug mode with auto-reload
- Changes to Python files are reflected immediately
- MongoDB data persists in Docker volumes

## License
MIT
