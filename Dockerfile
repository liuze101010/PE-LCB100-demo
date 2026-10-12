FROM python:3.12-slim
ARG INSTALL_EXTRAS=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd --create-home --uid 10001 polymer
WORKDIR /opt/polymer-agent
COPY . /opt/polymer-agent
RUN pip install ".${INSTALL_EXTRAS}"
RUN mkdir -p /opt/polymer-agent/.runtime && \
    chown -R polymer:polymer /opt/polymer-agent
USER polymer
EXPOSE 8765
CMD ["python", "-m", "uvicorn", "polymer_agent_demo.app:app", "--host", "0.0.0.0", "--port", "8765", "--proxy-headers"]
