# -------------------------------
# 1️⃣ Base Image
# -------------------------------
FROM python:3.11-slim AS base

# جلوگیری از سوال‌های نصب
ENV DEBIAN_FRONTEND=noninteractive

# -------------------------------
# 2️⃣ نصب وابستگی‌های سیستمی فقط یه بار (بدون مشکل libGL)
# -------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libzbar0 \
    libgl1-mesa-dev \
    libglx-mesa0 \
    libopencv-core-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# 3️⃣ تنظیم مسیر کاری پروژه
# -------------------------------
WORKDIR /app

# -------------------------------
# 4️⃣ نصب پکیج‌های پایتون
# -------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------
# 5️⃣ کپی کل پروژه
# -------------------------------
COPY . .

# -------------------------------
# 6️⃣ پورت و دستور اجرا
# -------------------------------
EXPOSE 8000
CMD ["python", "main.py"]
