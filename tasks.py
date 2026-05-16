import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from audio_stitcher import stitch_audio
from chunker import detect_chapters, smart_chunk
from config import OUTPUT_FOLDER, TEMP_FOLDER, TTS_MAX_WORKERS, USE_GPU
from file_processor import extract_text
from llm_cleaner import clean_with_diff
from models import Job
from tts_engine import TTSEngine
from video_generator import create_video


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _is_cancelled(job_id: str) -> bool:
    job = Job.get(job_id)
    return bool(job and job.data.get('cancelled', False))


def _fail(job_id: str, message: str) -> None:
    Job.update(job_id, {
        'status':   'failed',
        'message':  f'Error: {message}',
        'progress': 0,
    })


def _cleanup_wav_files(wav_pairs: list[tuple[int, str]]) -> None:
    for _, path in wav_pairs:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError as exc:
            print(f"  Could not remove temp file {path}: {exc}")


# ─── Stage 1 : Extract → Clean → Diff → Preview ───────────────────────────────

def process_file_task(job_id: str, app_context, active_threads: dict) -> None:
    """
    1. Extract text from the uploaded ebook
    2. Detect chapters (optional)
    3. Clean text with LLM  ← now reports per-chunk progress to DB
    4. Compute diff HTML for the preview highlight feature
    5. Save all data and set status = 'preview'
    """
    with app_context:
        job = Job.get(job_id)
        if not job:
            return

        try:
            if _is_cancelled(job_id):
                return

            # ── 1. Extract ────────────────────────────────────────────────────
            Job.update(job_id, {
                'status':   'extracting',
                'message':  'Extracting text from file…',
                'progress': 10,
            })

            full_text = extract_text(job.original_file)
            if not full_text or not full_text.strip():
                raise ValueError("Extracted text is empty — is the file readable?")

            print(f"  Extracted {len(full_text):,} chars from {job.original_filename}")
            Job.update(job_id, {
                'message':  f'Extracted {len(full_text):,} characters',
                'progress': 30,
            })

            # ── 2. Chapter detection ──────────────────────────────────────────
            if _is_cancelled(job_id):
                return

            chapters: list = []
            if job.detect_chapters:
                Job.update(job_id, {'message': 'Detecting chapters…', 'progress': 50})
                chapters = detect_chapters(full_text)
                Job.update(job_id, {'chapters': json.dumps(chapters)})
                print(f"  Found {len(chapters)} chapter(s)")

            # ── 3. LLM cleaning + diff ────────────────────────────────────────
            if _is_cancelled(job_id):
                return

            # Pre-calculate how many LLM chunks there will be so the UI shows
            # a meaningful total before the first chunk even finishes.
            from config import OPENROUTER_API_KEY
            llm_max_chars  = int(os.getenv('LLM_MAX_CHARS', '8000'))
            estimated_total = max(1, (len(full_text) + llm_max_chars - 1) // llm_max_chars)

            Job.update(job_id, {
                'status':       'cleaning',
                'message':      f'Cleaning text with AI… (0/{estimated_total} chunks)',
                'progress':     60,
                'chunks_done':  0,
                'total_chunks': estimated_total,
            })

            # ── Progress callback: fired after every LLM chunk ────────────────
            # Cleaning occupies progress 60 → 92.
            # The final save/diff step gets 92 → 100.
            def _llm_chunk_done(done: int, total: int) -> None:
                if _is_cancelled(job_id):
                    return
                pct = 60 + int((done / total) * 32)   # 60 % … 92 %
                Job.update(job_id, {
                    'chunks_done':  done,
                    'total_chunks': total,
                    'progress':     pct,
                    'message':      f'Cleaning chunk {done}/{total}…',
                })

            result = clean_with_diff(
                full_text,
                model_id=job.llm_model,
                custom_prompt=job.data.get('custom_prompt', ''),
                progress_callback=_llm_chunk_done,
            )

            cleaned   = result['cleaned']
            diff_html = result['diff_html']

            # ── 4. Save preview ───────────────────────────────────────────────
            if _is_cancelled(job_id):
                return

            Job.update(job_id, {
                'extracted_text': full_text,
                'cleaned_text':   cleaned,
                'diff_html':      diff_html,
                'status':         'preview',
                'message':        'Ready for preview and editing',
                'progress':       100,
                # Reset chunk counters so the TTS phase starts clean
                'chunks_done':    0,
                'total_chunks':   0,
            })

            print(f"  Job {job_id} ready for preview ({len(cleaned):,} chars cleaned)")

        except Exception as exc:
            if not _is_cancelled(job_id):
                _fail(job_id, str(exc))
            traceback.print_exc()
        finally:
            active_threads.pop(job_id, None)


# ─── Stage 2 : TTS + Stitch + Video ──────────────────────────────────────────

def start_tts_task(job_id: str, app_context, active_threads: dict) -> None:
    """
    1. Chunk the (edited) text
    2. Synthesise in parallel via Kokoro
    3. Stitch chunks into a single MP3
    4. Optionally generate an MP4 video
    """
    with app_context:
        job = Job.get(job_id)
        if not job:
            return

        wav_pairs: list[tuple[int, str]] = []

        try:
            if _is_cancelled(job_id):
                return

            text = job.edited_text or job.cleaned_text
            if not text or not text.strip():
                raise ValueError("No text to process — did you save the edited text?")

            # ── 1. Chunk ──────────────────────────────────────────────────────
            Job.update(job_id, {
                'status':   'tts',
                'message':  'Chunking text…',
                'progress': 10,
            })

            chapters: list = []
            if job.detect_chapters:
                chapters = detect_chapters(text)

            chunks = smart_chunk(
                text,
                max_chars=2000,
                with_chapters=job.detect_chapters,
                chapters=chapters,
            )
            if not chunks:
                raise RuntimeError("No text chunks created")

            total = len(chunks)
            print(f"  {total} chunk(s) ready for TTS")

            Job.update(job_id, {
                'total_chunks': total,
                'chunks_done':  0,
                'message':      f'Starting TTS for {total} chunk(s)…',
                'progress':     15,
            })

            # ── 2. Init TTS engine ────────────────────────────────────────────
            tts = TTSEngine(use_gpu=USE_GPU)

            # ── 3. Parallel synthesis ─────────────────────────────────────────
            failed_chunks: list[dict] = []

            def _process_chunk(idx: int, chunk_text: str) -> tuple[int, str | None, str | None]:
                if _is_cancelled(job_id):
                    return idx, None, 'Cancelled'
                try:
                    path = tts.synthesize(chunk_text, idx, job.voice, job.speed, job.pitch)
                    return idx, path, None
                except Exception as exc:
                    return idx, None, str(exc)

            print(f"  TTS workers: {TTS_MAX_WORKERS}")
            with ThreadPoolExecutor(max_workers=TTS_MAX_WORKERS) as executor:
                futures = {
                    executor.submit(
                        _process_chunk,
                        idx,
                        chunk['text'] if isinstance(chunk, dict) else chunk,
                    ): idx
                    for idx, chunk in enumerate(chunks)
                }

                done_count = 0
                for future in as_completed(futures):
                    if _is_cancelled(job_id):
                        executor.shutdown(wait=False, cancel_futures=True)
                        return

                    idx, wav_path, error = future.result()
                    if error and error != 'Cancelled':
                        failed_chunks.append({'chunk': idx, 'error': error})
                        print(f"  Chunk {idx} failed: {error[:80]}")
                        Job.update(job_id, {'failed_chunks': json.dumps(failed_chunks)})
                    elif wav_path:
                        wav_pairs.append((idx, wav_path))

                    done_count = len(wav_pairs)
                    progress   = 15 + int((done_count / total) * 70)
                    Job.update(job_id, {
                        'chunks_done': done_count,
                        'progress':    progress,
                        'message':     f'Synthesised {done_count}/{total} chunk(s)',
                    })

            if _is_cancelled(job_id):
                return

            if not wav_pairs:
                raise RuntimeError(f"No chunks synthesised ({len(failed_chunks)} failed)")

            print(f"  {len(wav_pairs)}/{total} chunks OK ({len(failed_chunks)} failed)")

            # ── 4. Stitch ─────────────────────────────────────────────────────
            Job.update(job_id, {'message': 'Stitching audio…', 'progress': 88})
            wav_pairs.sort(key=lambda x: x[0])
            final_audio = os.path.join(OUTPUT_FOLDER, f'{job_id}.mp3')
            stitch_audio([p for _, p in wav_pairs], final_audio)
            print(f"  Audio saved: {final_audio}")

            # ── 5. Video (optional) ───────────────────────────────────────────
            video_path: str | None = None
            if job.generate_video and not _is_cancelled(job_id):
                thumbnail = job.thumbnail
                if thumbnail and os.path.exists(thumbnail):
                    Job.update(job_id, {'message': 'Generating video…', 'progress': 93})
                    video_path = os.path.join(OUTPUT_FOLDER, f'{job_id}.mp4')
                    title = (
                        os.path.splitext(os.path.basename(job.original_file))[0]
                        .replace('_', ' ').replace('-', ' ').title()
                    )
                    try:
                        create_video(
                            final_audio, thumbnail, video_path,
                            title=title,
                            chapters=json.loads(job.data.get('chapters', '[]'))
                                     if job.data.get('chapters') else None,
                            enable_zoom=True,
                        )
                        print(f"  Video saved: {video_path}")
                    except Exception as vid_err:
                        print(f"  Video generation failed: {vid_err}")
                        video_path = None
                else:
                    print("  generate_video=True but no valid thumbnail — skipping")

            if _is_cancelled(job_id):
                return

            # ── 6. Done ───────────────────────────────────────────────────────
            done_msg = 'Processing complete!'
            if video_path:
                done_msg += ' Video generated.'
            if failed_chunks:
                done_msg += f' ({len(failed_chunks)} chunk(s) failed.)'

            Job.update(job_id, {
                'audio_file': final_audio,
                'video_file': video_path,
                'status':     'done',
                'message':    done_msg,
                'progress':   100,
            })
            print(f"  Job {job_id} complete")

        except Exception as exc:
            if not _is_cancelled(job_id):
                _fail(job_id, str(exc))
            print(f"  Job {job_id} failed: {exc}")
            traceback.print_exc()
        finally:
            _cleanup_wav_files(wav_pairs)
            active_threads.pop(job_id, None)


# ─── Cancellation ─────────────────────────────────────────────────────────────

def cancel_job_task(job_id: str, app_context, active_threads: dict) -> None:
    with app_context:
        Job.update(job_id, {
            'status':    'cancelled',
            'message':   'Job cancelled by user',
            'cancelled': True,
            'progress':  0,
        })
        active_threads.pop(job_id, None)
        print(f"  Job {job_id} cancelled")