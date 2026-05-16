import os
import threading
import traceback
from datetime import datetime

import shortuuid
from bson.objectid import ObjectId
from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for
from flask_pymongo import PyMongo
from PIL import Image

from config import (
    ALL_MODELS,
    DEFAULT_LLM_MODEL,
    KOKORO_AVAILABLE_VOICES,
    MAX_UPLOAD_BYTES,
    MONGO_DB,
    MONGO_URI,
    OUTPUT_FOLDER,
    SECRET_KEY,
    TEMP_FOLDER,
    UPLOAD_FOLDER,
    USE_GPU,
)
from models import Job, mongo
from tasks import cancel_job_task, process_file_task, start_tts_task


# ─── App setup ────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config['MONGO_URI']          = f'{MONGO_URI}{MONGO_DB}'
app.config['UPLOAD_FOLDER']      = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER']      = OUTPUT_FOLDER
app.config['SECRET_KEY']         = SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_BYTES

mongo.init_app(app)

active_threads: dict[str, threading.Thread] = {}

ALLOWED_EBOOK_EXT = {
    '.epub', '.mobi', '.azw', '.azw3', '.fb2',
    '.pdf',  '.rtf',  '.html', '.htm', '.docx',
    '.odt',  '.txt',
}
ALLOWED_IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.webp'}


# ─── Auto-start Kokoro ────────────────────────────────────────────────────────

def _start_kokoro_background() -> None:
    """
    Start the Kokoro-FastAPI server in a background thread so Flask can
    begin serving requests immediately while Kokoro warms up.
    """
    def _run():
        try:
            from kokoro_manager import start_kokoro
            mgr = get_manager()
            mgr.start()
            # ok = start_kokoro()
            # if ok:
            #     print("  Kokoro ready and healthy")
            # else:
            #     print("  Kokoro did not start — TTS will fail unless you start it manually")
        except Exception as exc:
            print(f"  Kokoro auto-start error: {exc}")

    t = threading.Thread(target=_run, daemon=True)
    t.start()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _ext(filename: str) -> str:
    return os.path.splitext(filename)[1].lower()


def save_upload(file) -> tuple[str, str]:
    ext = _ext(file.filename)
    if ext not in ALLOWED_EBOOK_EXT:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EBOOK_EXT))}"
        )
    timestamp   = datetime.now().strftime('%y%m%d')
    random_part = shortuuid.ShortUUID().random(length=6)
    job_id      = f'{timestamp}-{random_part}'
    filename    = f'{job_id}{ext}'
    path        = os.path.join(UPLOAD_FOLDER, filename)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file.save(path)
    print(f"  Saved: {file.filename} → {filename}  (job={job_id})")
    return job_id, path


def _save_thumbnail(thumb_file, job_id: str) -> str | None:
    if not thumb_file or not thumb_file.filename:
        return None
    if _ext(thumb_file.filename) not in ALLOWED_IMAGE_EXT:
        return None
    thumb_path = os.path.join(UPLOAD_FOLDER, f'{job_id}_thumb.jpg')
    try:
        img = Image.open(thumb_file)
        if img.mode in ('RGBA', 'LA', 'P'):
            bg = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            if img.mode in ('RGBA', 'LA'):
                bg.paste(img, mask=img.split()[-1])
                img = bg
            else:
                img = img.convert('RGB')
        else:
            img = img.convert('RGB')
        img.save(thumb_path, 'JPEG', quality=95)
        return thumb_path
    except Exception as exc:
        print(f"  Thumbnail error: {exc}")
        return None


def _start_thread(target, args: tuple) -> threading.Thread:
    t = threading.Thread(target=target, args=args, daemon=True)
    t.start()
    return t


# ─── HTML routes ──────────────────────────────────────────────────────────────

@app.route('/')
def index():
    try:
        active_jobs = Job.list_active()
    except Exception:
        active_jobs = []
    return render_template(
        'index.html',
        active_jobs=active_jobs,
        gemini_models=ALL_MODELS,
        available_voices=KOKORO_AVAILABLE_VOICES,
        use_gpu=USE_GPU,
    )


@app.route('/preview/<job_id>')
def preview_page(job_id: str):
    try:
        job = Job.get(job_id)
        if not job:
            return redirect(url_for('index'))
        job.data.setdefault('extracted_text', '')
        job.data.setdefault('cleaned_text', '')
        job.data.setdefault('diff_html', '')
        return render_template('preview.html', job=job)
    except Exception:
        traceback.print_exc()
        return redirect(url_for('index'))


@app.route('/status/<job_id>')
def status_page(job_id: str):
    try:
        job = Job.get(job_id)
        if not job:
            return redirect(url_for('index'))
        return render_template('status.html', job=job)
    except Exception:
        traceback.print_exc()
        return redirect(url_for('index'))


@app.route('/dashboard')
def dashboard():
    try:
        jobs = list(mongo.db.jobs.find().sort('created_at', -1).limit(100))
        return render_template('dashboard.html', jobs=jobs)
    except Exception:
        traceback.print_exc()
        return redirect(url_for('index'))


# ─── API — Upload ─────────────────────────────────────────────────────────────

@app.route('/api/upload-multiple', methods=['POST'])
def upload_multiple_files():
    files = request.files.getlist('files')
    if not files or not files[0].filename:
        return jsonify({'error': 'No files uploaded'}), 400

    llm_model    = request.form.get('llm_model', DEFAULT_LLM_MODEL)
    tts_engine   = request.form.get('tts_engine', 'kokoro')
    voice        = request.form.get('voice', 'default')
    custom_voice = request.form.get('custom_voice', '').strip()
    global_prompt = request.form.get('global_prompt', '')

    if custom_voice:
        voice = custom_voice

    try:
        speed = float(request.form.get('speed', 1.0))
    except (ValueError, TypeError):
        speed = 1.0
    try:
        pitch = float(request.form.get('pitch', 1.0))
    except (ValueError, TypeError):
        pitch = 1.0

    responses = []
    for idx, file in enumerate(files):
        if not file or not file.filename:
            continue
        try:
            job_id, path = save_upload(file)
            prompt          = request.form.get(f'prompt_{idx}', global_prompt)
            detect_chapters = request.form.get(f'detect_chapters_{idx}') == 'on'
            generate_video  = request.form.get(f'generate_video_{idx}') == 'on'
            thumbnail_path  = _save_thumbnail(request.files.get(f'thumbnail_{idx}'), job_id)

            job_data = {
                'original_file':     path,
                'original_filename': file.filename,
                'llm_model':         llm_model,
                'tts_engine':        tts_engine,
                'voice':             voice,
                'speed':             speed,
                'pitch':             pitch,
                'custom_prompt':     prompt,
                'generate_video':    generate_video,
                'detect_chapters':   detect_chapters,
                'thumbnail':         thumbnail_path,
                'cancelled':         False,
            }

            job = Job.create(job_data)
            t   = _start_thread(process_file_task, (job.id, app.app_context(), active_threads))
            active_threads[job.id] = t
            responses.append({'job_id': job.id, 'filename': file.filename})

        except ValueError as exc:
            responses.append({'filename': file.filename, 'error': str(exc)})
        except Exception as exc:
            traceback.print_exc()
            responses.append({'filename': file.filename, 'error': str(exc)})

    return jsonify({'jobs': responses}), 202


# ─── API — Job queries ────────────────────────────────────────────────────────

@app.route('/api/job/<job_id>', methods=['GET'])
def get_job(job_id: str):
    job = Job.get(job_id)
    if not job:
        return jsonify({'error': 'Not found'}), 404
    return jsonify(job.to_dict())


@app.route('/api/jobs/all', methods=['GET'])
def get_all_jobs():
    try:
        jobs   = list(mongo.db.jobs.find().sort('created_at', -1).limit(100))
        result = []
        for raw in jobs:
            j    = Job(raw)
            data = j.to_dict()
            data['filename'] = os.path.basename(raw.get('original_file', ''))
            result.append(data)
        return jsonify(result)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/jobs/active', methods=['GET'])
def get_active_jobs():
    try:
        jobs   = Job.list_active()
        result = []
        for raw in jobs:
            j    = Job(raw)
            data = j.to_dict()
            data['filename'] = os.path.basename(raw.get('original_file', ''))
            result.append(data)
        return jsonify(result)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/job/<job_id>/preview', methods=['GET'])
def preview_text(job_id: str):
    job = Job.get(job_id)
    if not job:
        return jsonify({'error': 'Not found'}), 404
    return jsonify({
        'text':      job.edited_text or job.cleaned_text,
        'diff_html': job.data.get('diff_html', ''),
    })


# ─── API — Edit & start TTS ───────────────────────────────────────────────────

@app.route('/api/job/<job_id>/edit', methods=['POST'])
def edit_text(job_id: str):
    job = Job.get(job_id)
    if not job:
        return jsonify({'error': 'Not found'}), 404

    data   = request.get_json(silent=True) or {}
    edited = data.get('text', '').strip()
    if not edited:
        return jsonify({'error': 'No text provided'}), 400

    Job.update(job_id, {
        'edited_text': edited,
        'status':      'preview_done',
        'message':     'Text saved — starting TTS',
    })

    t = _start_thread(start_tts_task, (job_id, app.app_context(), active_threads))
    active_threads[job_id] = t

    return jsonify({'status': 'ok', 'redirect': url_for('status_page', job_id=job_id)})


# ─── API — Download ───────────────────────────────────────────────────────────

@app.route('/api/job/<job_id>/output/<filetype>', methods=['GET'])
def download_output(job_id: str, filetype: str):
    job = Job.get(job_id)
    if not job:
        return jsonify({'error': 'Not found'}), 404

    if filetype == 'audio' and job.audio_file and os.path.exists(job.audio_file):
        return send_file(job.audio_file, as_attachment=True,
                         download_name=f'{job_id}.mp3', mimetype='audio/mpeg')

    if filetype == 'video' and job.video_file and os.path.exists(job.video_file):
        return send_file(job.video_file, as_attachment=True,
                         download_name=f'{job_id}.mp4', mimetype='video/mp4')

    return jsonify({'error': 'File not ready or does not exist'}), 404


# ─── API — Cancel ─────────────────────────────────────────────────────────────

@app.route('/api/job/<job_id>/cancel', methods=['POST'])
def cancel_job(job_id: str):
    job = Job.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if job.status in ('done', 'failed', 'cancelled'):
        return jsonify({'error': f'Job is already {job.status}'}), 400

    Job.update(job_id, {
        'status':    'cancelled',
        'message':   'Job cancelled by user',
        'cancelled': True,
        'progress':  0,
    })
    active_threads.pop(job_id, None)
    _purge_temp_chunks(job_id)
    return jsonify({'status': 'ok', 'message': 'Job cancelled'})


# ─── API — Delete ─────────────────────────────────────────────────────────────

@app.route('/api/job/<job_id>/delete', methods=['DELETE'])
def delete_job(job_id: str):
    job = Job.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    if job.status not in ('done', 'failed', 'cancelled'):
        Job.update(job_id, {'status': 'cancelled', 'message': 'Deleted by user', 'cancelled': True})
    active_threads.pop(job_id, None)

    for attr in ('original_file', 'audio_file', 'video_file', 'thumbnail'):
        _safe_remove(job.data.get(attr))
    _purge_temp_chunks(job_id)
    Job.delete(job_id)
    return jsonify({'status': 'ok', 'message': 'Job deleted'})


# ─── Utilities ────────────────────────────────────────────────────────────────

def _safe_remove(path: str | None) -> None:
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError as exc:
            print(f"  Could not delete {path}: {exc}")


def _purge_temp_chunks(job_id: str) -> None:
    try:
        for fname in os.listdir(TEMP_FOLDER):
            if job_id in fname:
                _safe_remove(os.path.join(TEMP_FOLDER, fname))
    except OSError:
        pass


# ─── Error handlers ───────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(413)
def request_too_large(error):
    mb = MAX_UPLOAD_BYTES // (1024 * 1024)
    return jsonify({'error': f'File too large. Maximum upload size is {mb} MB'}), 413

@app.errorhandler(500)
def internal_error(error):
    traceback.print_exc()
    return jsonify({'error': 'Internal server error'}), 500


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    for folder in (UPLOAD_FOLDER, OUTPUT_FOLDER, TEMP_FOLDER):
        os.makedirs(folder, exist_ok=True)


    from kokoro_manager import start_kokoro

    print("Starting Kokoro server...")

    if not start_kokoro():
        print("WARNING: Kokoro failed to start")
    else:
        print("Kokoro ready")
    # # Auto-start Kokoro (non-blocking)
    # _start_kokoro_background()

    app.run(host='0.0.0.0', port=5000, debug=False)
