import tornado.ioloop
import tornado.web
import tornado.websocket
from datetime import datetime
import os
from openai import OpenAI
import json

class TimeWebSocketHandler(tornado.websocket.WebSocketHandler):
    def initialize(self):
        """初始化对话历史"""
        self.query = [{"role": "system", "content": "You are a helpful assistant."}]
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def open(self):
        """客户端连接建立时触发"""
        print("WebSocket connected")
        self.send_time()  # 立即发送当前时间

    def on_message(self, message):
        """接收客户端消息"""
        try:
            print(f"Received: {message}")
            self.query.append({"role": "user", "content": message})  # 添加用户消息
            reply = self.chat_with_qwen()  # 调用 Qwen 模型
            self.write_message(reply)  # 直接返回模型回复（无需前缀 "Echo:"）
            self.query.append({"role": "assistant", "content": reply})  # 添加模型回复
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            self.write_message(error_msg)

    def chat_with_qwen(self):
        """与 Qwen 模型对话"""
        response = self.client.chat.completions.create(
            model="qwen-plus",
            messages=self.query[-10:]  # 限制上下文长度（避免内存问题）
        )
        return response.choices[0].message.content

    def send_time(self):
        """发送当前时间"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.write_message(json.dumps({"type": "time", "data": current_time}))

    def on_close(self):
        """连接关闭时触发"""
        print("WebSocket disconnected")

    def check_origin(self, origin):
        """允许跨域（仅开发环境使用）"""
        return True

def make_app():
    return tornado.web.Application([
        (r"/ws", TimeWebSocketHandler),
    ], websocket_ping_interval=20)

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    print("Server started at ws://localhost:8888/ws")
    tornado.ioloop.IOLoop.current().start()