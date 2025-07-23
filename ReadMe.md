# 🧠 AI Image Classifier with FastAPI

- This project is a deep learning image classifier served through a **FastAPI** backend and styled with **Bootstrap 5**. 
- It loads a pretrained **ResNet18** model from **PyTorch**, processes image uploads via **Pillow**, and returns top-1 predictions with confidence scores. 
- Includes user-friendly features like dark mode themes, real-time progress bar, and client-side image preview.

---

## 🚀 Features

- 📤 Upload any image for prediction
- 🧠 AI model: Pretrained ResNet18 (via torchvision)
- 📊 Confidence displayed with animated progress bar
- 🌙 Dark mode UI (separate themes for upload & results)
- 🖼 Image preview before submission
- ⏳ Loader while model is predicting
- ⚡ FastAPI backend with HTML frontend (Jinja2)

---

## 📦 Tech Stack

- **FastAPI** – backend framework
- **Uvicorn** – server for development
- **Torch + Torchvision** – pretrained ResNet model
- **Pillow** – image processing
- **Bootstrap 5** – UI design
- **Jinja2** – templating engine


---

## 🛠 Installation

1. **Clone the repo:**
```bash
 git clone https://github.com/harizonelopez/AI_FASTAPI-project.git
 cd AI_FASTAPI project
```

2. **Install dependencies**
```bash
 pip install -r requirements.txt
```

3. **Run the app**
```bash
 uvicorn app.main:app --reload
```

4. **Visit**
 Open your browser and go to

`http://127.0.0.1:8000`


## 👨‍💻 Author

- Made with 💙 by HarizoneLopez
- 📫 Reach me at: harizonelopez23@gmail.com


## 📄 License

- MIT License


