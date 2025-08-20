# 🔐 Security Guide - API Key Management

## ⚠️ CRITICAL SECURITY WARNINGS

**NEVER commit API keys, secrets, or sensitive configuration to version control!**
**NEVER share your API keys publicly!**
**NEVER hardcode secrets in your source code!**

## 🚀 Quick Start - Secure Configuration

### 1. Environment Setup

```bash
# Copy the example environment file
cp env.example .env

# Edit .env with your actual values
nano .env
```

### 2. Required API Keys

#### OpenAI API Key
```bash
# Get your API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY="sk-..."
```

#### Anthropic API Key
```bash
# Get your API key from: https://console.anthropic.com/
ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Generate Secure Secrets

```bash
# Generate secure keys for development
python -c "import secrets; print('SECRET_KEY:', secrets.token_urlsafe(32))"
python -c "import secrets; print('JWT_SECRET:', secrets.token_urlsafe(32))"
python -c "import secrets; print('ENCRYPTION_KEY:', secrets.token_urlsafe(32))"
```

## 🔑 API Key Management Best Practices

### ✅ DO:
- Use environment variables for all sensitive data
- Use `.env` files for local development
- Use secure secret management services in production
- Rotate API keys regularly
- Use least-privilege access for API keys
- Monitor API key usage and costs

### ❌ DON'T:
- Commit `.env` files to version control
- Hardcode API keys in source code
- Share API keys in public repositories
- Use the same API key across multiple projects
- Store API keys in plain text files
- Use default/example values in production

## 🛡️ Production Security Checklist

### Environment Variables
- [ ] All API keys are set via environment variables
- [ ] No hardcoded secrets in source code
- [ ] Production secrets are managed securely (e.g., AWS Secrets Manager, HashiCorp Vault)
- [ ] Database credentials are encrypted and secure

### Access Control
- [ ] API keys have minimal required permissions
- [ ] Database user has least-privilege access
- [ ] Redis access is restricted to application only
- [ ] File uploads are scanned for malware

### Monitoring & Logging
- [ ] API key usage is monitored and logged
- [ ] Failed authentication attempts are logged
- [ ] Cost monitoring is enabled for LLM APIs
- [ ] Security events are alerted on

## 🔧 Configuration Examples

### Development (.env)
```bash
ENVIRONMENT="development"
DEBUG=true
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="sk-ant-..."
DATABASE_URL="postgresql://user:pass@localhost:5432/dev_db"
```

### Production (Environment Variables)
```bash
export ENVIRONMENT="production"
export DEBUG=false
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export DATABASE_URL="postgresql://user:pass@prod-server:5432/prod_db"
```

### Docker Compose
```yaml
version: '3.8'
services:
  app:
    environment:
      - ENVIRONMENT=production
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - DATABASE_URL=${DATABASE_URL}
    env_file:
      - .env.production
```

## 🚨 Emergency Procedures

### If API Keys Are Compromised:

1. **Immediate Actions:**
   - Revoke the compromised API key immediately
   - Generate a new API key
   - Update all environment variables
   - Check for unauthorized usage

2. **Investigation:**
   - Review access logs
   - Check for data breaches
   - Audit all systems using the key
   - Update security documentation

3. **Recovery:**
   - Deploy new configuration
   - Monitor for any issues
   - Update incident response plan
   - Conduct security review

## 📚 Additional Resources

### Security Tools
- [OWASP Security Guidelines](https://owasp.org/)
- [Python Security Best Practices](https://python-security.readthedocs.io/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)

### Secret Management Services
- **AWS Secrets Manager** - For AWS environments
- **HashiCorp Vault** - Open-source secret management
- **Azure Key Vault** - For Azure environments
- **Google Secret Manager** - For GCP environments

### Monitoring Tools
- **OpenTelemetry** - Application monitoring
- **Prometheus** - Metrics collection
- **Grafana** - Visualization and alerting
- **ELK Stack** - Log analysis

## 🆘 Getting Help

If you encounter security issues:

1. **Don't panic** - Follow the emergency procedures
2. **Document everything** - Keep detailed logs
3. **Contact security team** - If available
4. **Report to relevant authorities** - If required by law
5. **Learn from the incident** - Update procedures

## 📝 Security Policy

This project follows a **zero-tolerance policy** for security violations. All contributors must:

- Follow security best practices
- Report security issues immediately
- Never commit sensitive data
- Use secure development practices
- Participate in security reviews

---

**Remember: Security is everyone's responsibility!** 🔐
