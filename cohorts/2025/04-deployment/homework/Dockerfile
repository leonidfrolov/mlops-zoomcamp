FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install -U pip & pip install pipenv numpy scipy

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "script.py", "script.py" ]

ENTRYPOINT [ "python", "script.py", "2023", "5" ]
