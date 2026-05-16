import json
import os
from datetime import datetime

from bson.objectid import ObjectId
from flask_pymongo import PyMongo

mongo = PyMongo()


class Job:
    """
    Wrapper around a MongoDB job document.

    Status lifecycle:
      uploaded → extracting → cleaning → preview → preview_done → tts → done
      (any stage can transition to: failed | cancelled)
    """

    def __init__(self, data: dict | None = None) -> None:
        self.data: dict = data or {}

    # ── Indexes ───────────────────────────────────────────────────────────────

    @staticmethod
    def ensure_indexes() -> None:
        try:
            mongo.db.jobs.create_index('status')
            mongo.db.jobs.create_index([('created_at', -1)])
        except Exception as exc:
            print(f"  Could not create indexes: {exc}")

    # ── CRUD ──────────────────────────────────────────────────────────────────

    @staticmethod
    def create(job_data: dict) -> 'Job':
        now = datetime.utcnow()
        job_data.update({
            'created_at':    now,
            'updated_at':    now,
            'status':        'uploaded',
            'progress':      0,
            'chunks_done':   0,
            'total_chunks':  0,
            'message':       'Uploaded successfully',
            'failed_chunks': [],
            'diff_html':     '',
        })
        result          = mongo.db.jobs.insert_one(job_data)
        job_data['_id'] = result.inserted_id
        print(f"  Job created: {result.inserted_id}")
        return Job(job_data)

    @staticmethod
    def get(job_id: str) -> 'Job | None':
        try:
            data = mongo.db.jobs.find_one({'_id': ObjectId(job_id)})
            return Job(data) if data else None
        except Exception as exc:
            print(f"  Job.get({job_id}): {exc}")
            return None

    @staticmethod
    def update(job_id: str, fields: dict) -> None:
        try:
            fields['updated_at'] = datetime.utcnow()
            mongo.db.jobs.update_one(
                {'_id': ObjectId(job_id)},
                {'$set': fields},
            )
        except Exception as exc:
            print(f"  Job.update({job_id}): {exc}")
            raise

    @staticmethod
    def delete(job_id: str) -> None:
        try:
            mongo.db.jobs.delete_one({'_id': ObjectId(job_id)})
            print(f"  Job deleted: {job_id}")
        except Exception as exc:
            print(f"  Job.delete({job_id}): {exc}")
            raise

    @staticmethod
    def list_all(limit: int = 100) -> list[dict]:
        try:
            return list(mongo.db.jobs.find().sort('created_at', -1).limit(limit))
        except Exception as exc:
            print(f"  Job.list_all: {exc}")
            return []

    @staticmethod
    def list_active(limit: int = 50) -> list[dict]:
        terminal = {'done', 'failed', 'cancelled'}
        try:
            return list(
                mongo.db.jobs.find(
                    {'status': {'$nin': list(terminal)}}
                ).sort('created_at', -1).limit(limit)
            )
        except Exception as exc:
            print(f"  Job.list_active: {exc}")
            return []

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        if not self.data:
            return {}
        data = self.data.copy()
        if '_id' in data:
            data['id'] = str(data.pop('_id'))
        for key in ('created_at', 'updated_at'):
            val = data.get(key)
            if val and hasattr(val, 'isoformat'):
                data[key] = val.isoformat()
        for key in ('chapters', 'failed_chunks'):
            val = data.get(key)
            if isinstance(val, str):
                try:
                    data[key] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    data[key] = []
        data.setdefault(
            'original_filename',
            os.path.basename(data.get('original_file', '')) or 'Unknown',
        )
        return data

    # ── Property accessors ────────────────────────────────────────────────────

    @property
    def id(self) -> str:
        return str(self.data.get('_id', ''))

    @property
    def status(self) -> str:
        return self.data.get('status', 'uploaded')

    @property
    def progress(self) -> int:
        return int(self.data.get('progress', 0))

    @property
    def message(self) -> str:
        return self.data.get('message', '')

    @property
    def cleaned_text(self) -> str:
        return self.data.get('cleaned_text', '')

    @property
    def edited_text(self) -> str:
        return self.data.get('edited_text', '')

    @property
    def extracted_text(self) -> str:
        return self.data.get('extracted_text', '')

    @property
    def diff_html(self) -> str:
        return self.data.get('diff_html', '')

    @property
    def audio_file(self) -> str:
        return self.data.get('audio_file', '')

    @property
    def video_file(self) -> str:
        return self.data.get('video_file', '')

    @property
    def total_chunks(self) -> int:
        return int(self.data.get('total_chunks', 0))

    @property
    def chunks_done(self) -> int:
        return int(self.data.get('chunks_done', 0))

    @property
    def llm_model(self) -> str:
        return self.data.get('llm_model', '')

    @property
    def tts_engine(self) -> str:
        return self.data.get('tts_engine', 'kokoro')

    @property
    def voice(self) -> str:
        return self.data.get('voice', 'default')

    @property
    def speed(self) -> float:
        return float(self.data.get('speed', 1.0))

    @property
    def pitch(self) -> float:
        return float(self.data.get('pitch', 1.0))

    @property
    def detect_chapters(self) -> bool:
        return bool(self.data.get('detect_chapters', False))

    @property
    def generate_video(self) -> bool:
        return bool(self.data.get('generate_video', False))

    @property
    def thumbnail(self) -> str:
        return self.data.get('thumbnail', '')

    @property
    def original_file(self) -> str:
        return self.data.get('original_file', '')

    @property
    def original_filename(self) -> str:
        stored = self.data.get('original_filename')
        if stored:
            return stored
        return os.path.basename(self.original_file) if self.original_file else 'Unknown'