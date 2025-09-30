# Render Deployment Configuration

This project is configured for deployment on Render with separate services for frontend and backend.

## Deployment Instructions

### 1. Backend Service (FastAPI)

**Service Type**: Web Service
**Environment**: Docker
**Build Command**: Automatic (uses Dockerfile)
**Start Command**: Automatic (defined in Dockerfile)

**Configuration**:
- **Name**: `clinical-mortality-backend`
- **Repository**: Connect your GitHub repository
- **Branch**: `main` or `master`
- **Root Directory**: `clinical-mortality-app/backend`
- **Environment**: Docker
- **Instance Type**: Starter (or higher based on needs)

**Environment Variables** (Required):
```
DATAIKU_API_URL=https://your-dataiku-instance.com/public/api/v1/predict
DATAIKU_API_TOKEN=your-dataiku-api-token
```

**Health Check**:
- **Path**: `/health`
- **Expected Response**: `{"status": "healthy"}`

### 2. Frontend Service (React/Nginx)

**Service Type**: Static Site
**Environment**: Docker
**Build Command**: Automatic (uses Dockerfile)

**Configuration**:
- **Name**: `clinical-mortality-frontend`
- **Repository**: Connect your GitHub repository
- **Branch**: `main` or `master`
- **Root Directory**: `clinical-mortality-app/frontend`
- **Environment**: Docker
- **Instance Type**: Starter

**Build Settings**:
- **Build Command**: `npm run build`
- **Publish Directory**: `dist`

### 3. Auto-Deploy Settings

- ✅ **Auto-Deploy**: Enable for `main`/`master` branch
- ✅ **Pull Request Previews**: Enable (optional)

## 🔧 Environment Variables

### Backend Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DATAIKU_API_URL` | URL to your Dataiku ML model API endpoint | Yes |
| `DATAIKU_API_TOKEN` | Authentication token for Dataiku API | Yes |

### Frontend Environment Variables (if needed)

| Variable | Description | Required |
|----------|-------------|----------|
| `VITE_API_URL` | Backend API URL (auto-configured by Render) | No |

## 📡 Service Communication

The frontend will automatically communicate with the backend using Render's internal networking. Update the frontend API calls to use the backend service URL provided by Render.

## 🔗 Custom Domains (Optional)

After deployment, you can configure custom domains:
- **Frontend**: `https://your-app-name.com`
- **Backend API**: `https://api.your-app-name.com`

## 📊 Monitoring

Render provides built-in monitoring:
- **Logs**: Real-time application logs
- **Metrics**: CPU, Memory, and Network usage
- **Health Checks**: Automatic service health monitoring
- **Alerts**: Email notifications for service issues

## 🔒 Security

- All communications use HTTPS by default
- Environment variables are encrypted
- Regular security updates for the underlying infrastructure
- DDoS protection included

## 💰 Pricing Considerations

- **Starter instances**: $7/month per service
- **Standard instances**: $25/month per service
- **Static sites**: Free tier available
- **Database**: PostgreSQL available if needed

## 🚀 Deployment Steps

1. **Connect Repository**:
   - Link your GitHub repository to Render
   - Select the appropriate branch (`main` or `master`)

2. **Configure Backend Service**:
   - Create new Web Service
   - Set root directory to `clinical-mortality-app/backend`
   - Configure environment variables
   - Deploy

3. **Configure Frontend Service**:
   - Create new Static Site
   - Set root directory to `clinical-mortality-app/frontend`
   - Deploy

4. **Test Deployment**:
   - Verify health checks pass
   - Test API endpoints
   - Verify frontend loads correctly

5. **Configure Domain** (Optional):
   - Set up custom domains
   - Configure DNS settings

## 📝 Notes

- The CI/CD pipeline ensures code quality before deployment
- Docker builds are optimized for production
- Services will auto-restart on failure
- Render provides automatic SSL certificates
- Database backups available for paid plans

## 🆘 Troubleshooting

Common issues and solutions:

1. **Build Failures**:
   - Check build logs in Render dashboard
   - Verify Dockerfile syntax
   - Ensure all dependencies are listed

2. **Environment Variables**:
   - Verify all required variables are set
   - Check variable names match exactly
   - Ensure sensitive data is not exposed

3. **Service Communication**:
   - Update frontend API URLs to use Render service URLs
   - Check CORS settings in backend
   - Verify network policies

For more help, check the [Render Documentation](https://render.com/docs).