# Builder stage
FROM paddlepaddle/paddle:2.6.1

LABEL org.opencontainers.image.source https://github.com/BRM-Teck/d2s_core

# Create directory for the app user
RUN mkdir -p /home/app && addgroup --system app && adduser --system --group app

# Set environment variables
ENV HOME=/home/app
ENV APP_HOME=/home/app/web

# Create necessary directories
RUN mkdir $APP_HOME

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir -r requirements.txt


# Copy project source code
COPY src/ $APP_HOME

# Change ownership to the app user
RUN chown -R app:app $APP_HOME

# Switch to the app user
# Set working directory
WORKDIR $APP_HOME
USER app

# Set entrypoint
CMD ["python", "d2s_api.py"]
