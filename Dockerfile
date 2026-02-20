FROM python:3.10.13-slim

WORKDIR /app

# Install required system libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements-render.txt

EXPOSE 10000

CMD ["streamlit", "run", "src/ceras/streamlit_app.py", "--server.port=10000", "--server.address=0.0.0.0"]
