FROM python:3.11-slim

WORKDIR /app

# env
# prevents Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE 1
# prevents Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED 1

# dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# project
COPY . .

# run
CMD ["streamlit", "run", "app.py"]
