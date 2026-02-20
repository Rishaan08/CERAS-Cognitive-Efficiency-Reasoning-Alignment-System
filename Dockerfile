FROM python:3.10.13-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements-render.txt

EXPOSE 10000

CMD ["streamlit", "run", "src/ceras/streamlit_app.py", "--server.port=10000", "--server.address=0.0.0.0"]