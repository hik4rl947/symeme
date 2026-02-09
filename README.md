# 对称图像生成器 (Flask)

这是一个基于 Flask 的小工具：上传图片后，可以选择分割比例，将左侧或右侧区域镜像到另一侧并生成对称图像。

### 依赖

使用 pip 安装：

```bash
pip install -r requirements.txt
```

### 运行（开发模式）

在 Windows PowerShell:

```powershell
set FLASK_APP=app.py
set FLASK_ENV=development
flask run
```

或在 CMD:

```cmd
set FLASK_APP=app.py
set FLASK_ENV=development
flask run
```

打开浏览器访问 `http://127.0.0.1:5000/`，上传图片，选择分割比例与镜像方向（左→右 或 右→左），页面会显示左/右原始区域、镜像预览及带分割线的合成结果，可直接下载 PNG。
