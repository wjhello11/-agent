FROM ghcr.nju.edu.cn/xinnan-tech/xiaozhi-esp32-server:server_latest

WORKDIR /opt/xiaozhi-esp32-server

# Copy the local source tree into the runtime image so local code changes
# such as clinical_ltm and connection hooks are included in the container.
COPY . /opt/xiaozhi-esp32-server

RUN python -m pip install --no-cache-dir pypdf==6.10.0 python-docx==1.2.0

RUN mkdir -p /opt/xiaozhi-esp32-server/data /opt/xiaozhi-esp32-server/tmp

CMD ["python", "app.py"]
