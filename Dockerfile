FROM python:3.12-slim

WORKDIR /workspace

# Build artifacts are generated at container start (see entrypoint) unless
# they are mounted in, so the image itself stays data-free.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY static/ static/
COPY tests/ tests/
COPY "Gabungan_Surabaya_Sidoarjo.csv" "Data Sekolah - Copy of Data FULL Surabaya.csv" "Data Sekolah - Sidoarjo FULL.csv" ./

ENV APP_HOST=0.0.0.0 APP_PORT=8000
EXPOSE 8000

# build_db -> build_index -> train -> serve
CMD ["sh", "-c", "python -m app.build_db && python -m app.build_index && python -m app.predict_spp --train && uvicorn app.main:app --host $APP_HOST --port $APP_PORT"]
