FROM node:18-alpine

WORKDIR /app

# Copier package.json et package-lock.json (si disponible)
COPY package*.json ./

# Installer les dépendances
RUN npm install

# Copier le code source
COPY . .

# Construire l'application pour la production
RUN npm run build

# Utiliser nginx pour servir l'application
FROM nginx:alpine
COPY --from=0 /app/dist /usr/share/nginx/html

# Copier la configuration nginx personnalisée
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]