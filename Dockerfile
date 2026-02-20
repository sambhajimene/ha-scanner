FROM python:3.11

WORKDIR /app

# ✅ Install timezone data
RUN apt-get update && apt-get install -y tzdata && rm -rf /var/lib/apt/lists/*

# ✅ Set Indian Timezone
ENV TZ=Asia/Kolkata
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY requirements.txt .

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create JSON with write permissions
RUN touch /app/live_ha_signals.json && chmod 666 /app/live_ha_signals.json

EXPOSE 8501

ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501

CMD ["streamlit", "run", "ha_dashboard.py"]

##======================================================================
# FROM python:3.11

# WORKDIR /app

# COPY requirements.txt .

# RUN python -m pip install --upgrade pip
# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# EXPOSE 8501

# ENV STREAMLIT_SERVER_HEADLESS=true
# ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
# ENV STREAMLIT_SERVER_PORT=8501

# CMD ["streamlit", "run", "ha_dashboard.py"]
