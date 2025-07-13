# استفاده از image پایه پایتون ۳.۹
FROM python:3.9-slim

# تنظیم دایرکتوری کاری درون کانتینر
WORKDIR /app

# کپی فایل requirements.txt به داخل کانتینر
COPY requirements2.txt .

# نصب وابستگی‌های پایتون
RUN pip install --no-cache-dir -r requirements2.txt

# کپی تمام فایل‌های پروژه به داخل کانتینر
COPY . .

# دستور اجرای برنامه
CMD ["python", "index.py"]