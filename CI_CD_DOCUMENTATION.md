# CI/CD Pipeline Documentation

This document describes the CI/CD pipeline setup for the Clinical Mortality Prediction application, designed to ensure code quality and seamless deployment to Render.

## Overview

The CI/CD pipeline consists of two main workflows:

1. **CI/CD Pipeline** (`ci-cd.yml`) - Runs on PRs to `develop`/`master`/`main`
2. **Deploy to Render** (`deploy.yml`) - Runs on pushes to `main`/`master`

## Pipeline Components

### 1. Code Quality (`code-quality` job)

**Python Backend Checks:**
- **Black**: Code formatting
- **isort**: Import sorting
- **Flake8**: Linting and style guide enforcement
- **Bandit**: Security vulnerability scanning
- **Safety**: Dependency vulnerability checking
- **MyPy**: Static type checking

### 2. Frontend Quality (`frontend-quality` job)

**React/JavaScript Checks:**
- **ESLint**: JavaScript/React linting
- **Prettier**: Code formatting
- **Build**: Production build verification

### 3. Backend Tests (`backend-tests` job)

**Test Coverage:**
- **Pytest**: Unit and integration tests
- **Coverage**: Code coverage reporting

### 4. Docker Build (`docker-build` job)

**Container Building:**
- **Docker Build**: Multi-architecture builds
- **Image Testing**: Verify containers build successfully

### 5. Integration Tests (`integration-tests` job)

**Full Stack Testing:**
- **Docker Compose**: Full application stack
- **Health Checks**: API and frontend availability
- **API Testing**: Endpoint functionality

### 6. Deployment Ready (`deployment-ready` job)

**Production Readiness:**
- ✅ **Quality Gates**: All checks passed
- 📦 **Artifacts**: Build artifacts ready
- 📊 **Summary**: Deployment readiness report

## 🔧 Configuration Files

### Code Quality Configuration

**Python (`pyproject.toml`):**
```toml
[tool.black]
line-length = 88
target-version = ['py311']

[tool.isort]
profile = "black"
line_length = 88

[tool.mypy]
python_version = "3.11"
disallow_untyped_defs = true
```

**Flake8 (`.flake8`):**
```ini
[flake8]
max-line-length = 88
extend-ignore = E203, W503, E501
max-complexity = 10
```

**Frontend (`.eslintrc.json`):**
```json
{
  "extends": ["react-app", "react-app/jest"],
  "rules": {
    "no-unused-vars": "warn",
    "no-console": "warn",
    "prefer-const": "error"
  }
}
```

**Prettier (`.prettierrc`):**
```json
{
  "semi": true,
  "trailingComma": "es5",
  "singleQuote": true,
  "printWidth": 80
}
```

## 🚀 Deployment Pipeline

### Render Deployment Features

**Automatic Deployment:**
- ✅ Triggered on push to `main`/`master`
- 🔍 Change detection for services
- 🏥 Health checks post-deployment
- 📊 Deployment status reporting

**Service Configuration:**
- **Backend**: Web Service with Docker
- **Frontend**: Static Site with Docker
- **Auto-scaling**: Based on traffic
- **SSL**: Automatic HTTPS certificates

### Environment Variables

**Required for Backend:**
```env
DATAIKU_API_URL=https://your-dataiku-instance.com/api/v1/predict
DATAIKU_API_TOKEN=your-api-token
```

## 📊 Quality Gates

### Pull Request Requirements

Before merging to `develop` or `master`, all PRs must pass:

1. ✅ **Code Quality**: All linting and formatting checks
2. ✅ **Security**: No high/critical vulnerabilities
3. ✅ **Tests**: All tests passing with >80% coverage
4. ✅ **Build**: Successful Docker builds
5. ✅ **Integration**: End-to-end tests passing

### Branch Protection Rules

**Recommended GitHub Settings:**
```yaml
Branch Protection Rules:
  - Require status checks to pass before merging
  - Require branches to be up to date before merging
  - Require review from code owners
  - Dismiss stale PR approvals when new commits are pushed
  - Require linear history
```

## 🔒 Security Features

### Container Security
- 🔍 **Trivy Scanning**: Container vulnerability detection
- 👤 **Non-root Users**: Containers run with limited privileges
- 🛡️ **Minimal Images**: Alpine/slim base images
- 🔒 **Security Headers**: Nginx security configuration

### Dependency Management
- 📦 **Dependabot**: Automated dependency updates
- 🛡️ **Safety**: Python package vulnerability scanning
- 🔍 **Audit**: npm audit for JavaScript packages
- 📊 **SARIF**: Security findings in GitHub Security tab

## 📈 Monitoring & Observability

### GitHub Insights
- 📊 **Action Runs**: Pipeline execution history
- 🔍 **Security**: Vulnerability alerts and reports
- 📈 **Coverage**: Test coverage trends
- 📦 **Dependencies**: Dependency graph and updates

### Render Monitoring
- 📊 **Metrics**: CPU, memory, network usage
- 📝 **Logs**: Real-time application logs
- 🚨 **Alerts**: Service health notifications
- 📈 **Analytics**: Request metrics and performance

## 🛠️ Local Development

### Code Quality Setup

**Install tools:**
```bash
# Python tools
pip install black isort flake8 bandit safety mypy pytest pytest-cov

# Frontend tools
cd clinical-mortality-app/frontend
npm install eslint prettier
```

**Pre-commit hooks (recommended):**
```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml and run
pre-commit install
```

### Testing Locally

**Backend:**
```bash
cd clinical-mortality-app/backend
pytest tests/ -v --cov=.
```

**Frontend:**
```bash
cd clinical-mortality-app/frontend
npm run lint
npm run format:check
npm run build
```

**Docker:**
```bash
# Test full stack
docker-compose up --build
```

## 🔧 Troubleshooting

### Common Issues

**1. Build Failures:**
- Check Dockerfile syntax
- Verify all dependencies are listed
- Ensure base images are available

**2. Test Failures:**
- Check test environment variables
- Verify mock data and fixtures
- Ensure database connections (if any)

**3. Security Scan Failures:**
- Update vulnerable dependencies
- Review security findings
- Apply security patches

**4. Deployment Issues:**
- Verify environment variables in Render
- Check service logs in Render dashboard
- Validate health check endpoints

### Getting Help

1. **GitHub Issues**: Create issues for bugs or feature requests
2. **Discussions**: Use GitHub Discussions for questions
3. **Logs**: Check workflow logs in GitHub Actions
4. **Render Support**: Use Render dashboard for deployment issues

## 📚 References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Render Documentation](https://render.com/docs)
- [Docker Best Practices](https://docs.docker.com/develop/best-practices/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [React Deployment](https://create-react-app.dev/docs/deployment/)

---

**Last Updated**: September 2025  
**Maintained By**: MLOps Team - EPITA 2025