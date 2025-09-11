FROM public.ecr.aws/docker/library/python:3.13-slim
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter
ENV PORT=8000
WORKDIR /var/task
COPY requirements.txt ./
RUN python -m pip install -r requirements.txt
COPY *.py ./
CMD exec uvicorn --host 0.0.0.0 --port=$PORT simple_mcp_calculator:app
