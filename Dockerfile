FROM python:3.9-slim-buster

RUN apt update && \
    apt install -y curl && \
    curl -sL https://deb.nodesource.com/setup_14.x | bash - && \
    apt install -y nodejs

# Set the working directory for the client
WORKDIR /app

# Copy only the client directory and package files for caching purposes
COPY client/ ./
COPY . /app
RUN npm install


# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 4100

# Define environment variable for additional configuration
ENV NAME MioGatto

# Run the application with the specific argument
CMD ["python", "-m", "server", "2107.10832-gpt-4"]

