# Build stage
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci || npm install

# Copy source code
COPY . .

# Build argument for API URL
ARG VITE_API_URL=https://ml-mortality-prediction-with-mlops.onrender.com
ENV VITE_API_URL=$VITE_API_URL

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built app
COPY --from=builder /app/dist /usr/share/nginx/html

# Build argument to determine which nginx config to use
ARG ENVIRONMENT=production

# Copy appropriate nginx config based on environment
COPY nginx.conf /etc/nginx/conf.d/default.conf.prod
COPY nginx.local.conf /etc/nginx/conf.d/default.conf.local

# Use the appropriate config file based on VITE_API_URL
RUN if echo "$VITE_API_URL" | grep -q "localhost\|backend"; then \
        cp /etc/nginx/conf.d/default.conf.local /etc/nginx/conf.d/default.conf; \
    else \
        cp /etc/nginx/conf.d/default.conf.prod /etc/nginx/conf.d/default.conf; \
    fi

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]