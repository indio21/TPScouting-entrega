FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

RUN python -m pip install --no-cache-dir --upgrade \
        pip==26.2.1 \
        setuptools==83.0.0

COPY requirements-lock.txt requirements-torch-cpu.txt ./
RUN python -m pip install --no-cache-dir -r requirements-lock.txt \
    && python -m pip install --no-cache-dir -r requirements-torch-cpu.txt \
    && python -m pip check

COPY scouting_app/ ./scouting_app/

RUN addgroup --system tpscouting \
    && adduser --system --ingroup tpscouting --home /app tpscouting \
    && chown -R tpscouting:tpscouting /app

USER tpscouting

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/health', timeout=3)" || exit 1

CMD ["sh", "-c", "python scouting_app/seed_demo_data.py && exec gunicorn app:app --chdir scouting_app --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:${PORT}"]
