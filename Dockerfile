FROM python:3.6-slim
COPY ./app.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./lgbm_model.txt /deploy/
WORKDIR /deploy/
RUN 
RUN pip install -r requirements.txt && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        build-essential \
        gcc \
        g++ \
        curl \
        git && \
    git clone --recursive --branch stable --depth 1 https://github.com/Microsoft/LightGBM && \
    cd LightGBM/python-package && python setup.py install
EXPOSE 5000
# Start the app
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]