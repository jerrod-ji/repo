FROM python:3.9-slim
# 设置工作目录
WORKDIR /agent
# 复制当前目录下的所有文件到容器的工作目录
COPY . /agent
# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt
# 设置环境变量
ENV LANG C.UTF-8
# 设置容器启动时执行的命令
CMD ["python", "main.py"]
# 暴露端口
EXPOSE 8000