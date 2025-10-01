FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV CSV_PATH="data/All_Diets.csv" \
    OUT_DIR="outputs" \
    TOPN="5" \
    SHOW="0"

CMD ["sh", "-c", "python data_analysis.py --csv \"$CSV_PATH\" --out \"$OUT_DIR\" --topn \"$TOPN\" --show \"$SHOW\""]
