FROM python:3.7

WORKDIR /intentclassification

COPY dependencies.txt .
RUN pip install --no-cache-dir -r dependencies.txt

COPY . .
CMD ["python", "-m", "scripts.api.api"]
