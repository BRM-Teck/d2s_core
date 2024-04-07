# Builder stage
FROM registry.baidubce.com/paddlepaddle/paddle:2.6.1

LABEL org.opencontainers.image.source https://github.com/BRM-Teck/Doc2Struct_web


# Set working directory
WORKDIR /usr/src/app

# Install system dependencies
# RUN apk update && \
#     apk add --no-cache gcc musl-dev

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /usr/src/app/wheels -r requirements.txt



ARG APP_DATA_DIR
ARG APP_LOGGING_FILE_DIR

ENV APP_DATA_DIR=$APP_DATA_DIR
ENV APP_LOGGING_FILE_DIR=$APP_LOGGING_FILE_DIR

# Create directory for the app user
RUN mkdir -p /home/app && \
    addgroup -S app && \
    adduser -S -G app app

# Set environment variables
ENV HOME=/home/app
ENV APP_HOME=/home/app/web

# Create necessary directories
RUN mkdir $APP_HOME $APP_HOME/static $APP_HOME/media $APP_DATA_DIR $APP_LOGGING_FILE_DIR

# Install netcat for healthcheck
RUN apk add --no-cache netcat-openbsd poppler-utils

# Copy wheels and requirements.txt from builder stage
COPY --from=builder /usr/src/app/wheels /wheels
COPY --from=builder /usr/src/app/requirements.txt .

# Install Python dependencies from wheels
RUN pip install --no-cache /wheels/*

# Copy entrypoint script
COPY bin/entrypoint.sh $APP_HOME/
RUN chmod +x $APP_HOME/entrypoint.sh

# Copy project source code
COPY src/ $APP_HOME

# Change ownership to the app user
RUN chown -R app:app $APP_HOME
RUN chown -R app:app $APP_LOGGING_FILE_DIR $APP_DATA_DIR

# Switch to the app user
# Set working directory
WORKDIR $APP_HOME
USER app

# Set entrypoint
ENTRYPOINT ["/home/app/web/entrypoint.sh"]
