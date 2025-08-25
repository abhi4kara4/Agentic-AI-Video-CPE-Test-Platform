# Frontend Dockerfile for AI Video Test Platform
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY frontend/package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY frontend/ ./

# Expose port
EXPOSE 3000

# Start Vite development server
CMD ["npm", "start"]