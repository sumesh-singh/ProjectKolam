# Kolam AI System - Deployment Guide

This guide provides comprehensive instructions for deploying the Kolam AI Recognition and Recreation System.

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for development)
- 8GB+ RAM recommended
- 4GB+ free disk space

### One-Command Deployment

```bash
# Make deployment script executable
chmod +x backend/deploy.sh

# Deploy the system
./backend/deploy.sh
```

The system will be available at: http://localhost:8000

## 📋 Deployment Options

### Option 1: Docker Compose (Recommended)

```bash
cd backend
docker-compose up -d
```

### Option 2: Manual Docker Build

```bash
cd backend
docker build -t kolam-ai:latest .
docker run -p 8000:8000 -v $(pwd)/backend/models:/app/backend/models:ro kolam-ai:latest
```

### Option 3: Development Mode

```bash
cd backend
call env/Scripts/activate.bat  # Windows
# source env/bin/activate     # Linux/Mac

pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# Application Configuration
APP_NAME=KolamAI
DEBUG=False
API_V1_PREFIX=/api/v1

# Model Configuration
MODEL_PATH=/app/backend/models
MAX_BATCH_SIZE=32
ENABLE_GPU=False

# Security
SECRET_KEY=your-secret-key-here
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8080"]

# Performance
WORKERS=1
MAX_REQUEST_SIZE=10485760  # 10MB
```

### Model Files

Ensure the following model files exist in `backend/models/`:

- `kolam_cnn_final.h5` - Trained CNN model for pattern recognition
- `kolam_generator.h5` - Trained GAN generator for pattern creation
- `kolam_discriminator.h5` - Trained GAN discriminator
- `training_history.json` - Training history and metadata

## 📡 API Endpoints

### Base URL: `http://localhost:8000/api/v1`

### Health Check

```http
GET /health
```

### Pattern Recognition

```http
POST /ai/recognize
Content-Type: multipart/form-data

# Request body: image file
```

**Response:**

```json
{
  "predicted_class": 5,
  "predicted_label": "south_india_tamil_nadu",
  "confidence": 0.8765,
  "confidence_percentage": 87.65,
  "probabilities": [0.001, 0.002, ..., 0.8765, ...],
  "timestamp": "2025-09-29T10:30:00.000Z"
}
```

### Pattern Generation

```http
POST /ai/generate
Content-Type: application/json

{
  "num_patterns": 3,
  "class_condition": null
}
```

**Response:**

```json
{
  "patterns": [
    {
      "pattern_id": "generated_20250929_103000_0",
      "image_shape": [128, 128, 3],
      "generation_timestamp": "2025-09-29T10:30:00.000Z",
      "generation_method": "gan",
      "class_condition": null
    }
  ],
  "num_generated": 3,
  "generation_timestamp": "2025-09-29T10:30:00.000Z",
  "model_info": {
    "generator_params": 2345571,
    "latent_dimension": 100
  }
}
```

### Complete Workflow

```http
POST /ai/analyze_and_generate
Content-Type: multipart/form-data

# Request body: image file + num_variations parameter
```

## 🧪 Testing the Deployment

### 1. Health Check

```bash
curl http://localhost:8000/health
```

### 2. API Documentation

Visit: http://localhost:8000/docs

### 3. Test Pattern Generation

```bash
curl -X POST 'http://localhost:8000/api/v1/ai/generate' \
     -H 'Content-Type: application/json' \
     -d '{"num_patterns": 2}'
```

### 4. Test Pattern Recognition

```bash
curl -X POST 'http://localhost:8000/api/v1/ai/recognize' \
     -F 'image=@sample_kolam.jpg'
```

## 📊 Monitoring

### Application Logs

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f kolam-ai-api
```

### Performance Metrics

The application includes built-in performance monitoring:

- Model inference time
- Memory usage
- Request count and latency
- Error rates

### Health Checks

Automatic health checks every 30 seconds:

- Model availability
- Memory usage
- Response time
- Error status

## 🔧 Troubleshooting

### Common Issues

1. **Models not loading**

   ```bash
   # Check if model files exist
   ls -la backend/models/

   # Verify model integrity
   python -c "from kolam_cnn_model import KolamCNNModel; m = KolamCNNModel(); m.load_model('backend/models/kolam_cnn_final.h5'); print('Model loaded successfully')"
   ```

2. **Memory issues**

   ```bash
   # Reduce batch size in environment variables
   echo "MAX_BATCH_SIZE=16" >> .env

   # Enable GPU if available
   echo "ENABLE_GPU=True" >> .env
   ```

3. **Port conflicts**

   ```bash
   # Change port in docker-compose.yml
   # Or stop conflicting services
   docker-compose down
   ```

4. **Slow inference**
   - Enable GPU acceleration
   - Reduce model input size
   - Use model quantization

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🚀 Production Deployment

### Cloud Deployment (AWS/GCP/Azure)

1. **Build optimized Docker image:**

   ```dockerfile
   FROM python:3.11-slim

   # Install only production dependencies
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   # Copy application code
   COPY . .

   # Use production server
   CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. **Set up reverse proxy (Nginx):**

   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

3. **Configure monitoring:**
   - Set up Prometheus metrics
   - Configure log aggregation
   - Set up alerts for model performance

### Scalability

For high-traffic scenarios:

1. **Horizontal scaling:**

   ```bash
   docker-compose up -d --scale kolam-ai-api=3
   ```

2. **Load balancing:**

   - Use Kubernetes
   - Set up auto-scaling based on CPU/memory
   - Implement model caching

3. **Performance optimization:**
   - Enable GPU acceleration
   - Use model quantization
   - Implement request batching

## 🔒 Security

### Best Practices

1. **API Security:**

   - Implement rate limiting
   - Add authentication/authorization
   - Validate input file sizes and types

2. **Model Security:**

   - Scan models for vulnerabilities
   - Implement model versioning
   - Set up model update procedures

3. **Infrastructure Security:**
   - Use HTTPS in production
   - Implement firewall rules
   - Regular security updates

## 📈 Performance Optimization

### GPU Acceleration

```python
# Enable GPU in environment
echo "ENABLE_GPU=True" >> .env
echo "CUDA_VISIBLE_DEVICES=0" >> .env
```

### Model Optimization

1. **Quantization:**

   ```python
   # Convert models to quantized format
   python -c "import tensorflow as tf; converter = tf.lite.TFLiteConverter.from_saved_model('models/'); converter.optimizations = [tf.lite.Optimize.DEFAULT]; tflite_model = converter.convert()"
   ```

2. **Caching:**
   - Implement Redis caching for frequent requests
   - Cache model predictions
   - Use CDN for generated images

### Monitoring

Set up monitoring with:

- **Prometheus** for metrics collection
- **Grafana** for visualization
- **AlertManager** for alerting

## 🤝 Support

### Getting Help

1. **Check logs:** `docker-compose logs kolam-ai-api`
2. **Review documentation:** See `README_KOLAM_AI.md`
3. **Test endpoints:** Use the interactive API docs at `/docs`
4. **Check system resources:** Monitor CPU, memory, and disk usage

### Common Solutions

- **Out of memory:** Reduce batch size or enable GPU
- **Slow inference:** Check model loading and input preprocessing
- **Import errors:** Verify all dependencies are installed
- **File not found:** Check file paths and permissions

## 📝 API Examples

### Python Client Example

```python
import requests

# Pattern recognition
with open('kolam_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/ai/recognize',
        files={'image': f}
    )
    result = response.json()
    print(f"Predicted: {result['predicted_label']} ({result['confidence_percentage']:.1f}%)")

# Pattern generation
response = requests.post(
    'http://localhost:8000/api/v1/ai/generate',
    json={'num_patterns': 3}
)
patterns = response.json()
print(f"Generated {patterns['num_generated']} patterns")
```

### JavaScript Client Example

```javascript
// Pattern recognition
const formData = new FormData();
formData.append("image", fileInput.files[0]);

fetch("http://localhost:8000/api/v1/ai/recognize", {
  method: "POST",
  body: formData,
})
  .then((response) => response.json())
  .then((data) => {
    console.log(
      `Predicted: ${data.predicted_label} (${data.confidence_percentage.toFixed(
        1
      )}%)`
    );
  });

// Pattern generation
fetch("http://localhost:8000/api/v1/ai/generate", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    num_patterns: 3,
  }),
})
  .then((response) => response.json())
  .then((data) => {
    console.log(`Generated ${data.num_generated} patterns`);
  });
```

---

**Deployment Status:** ✅ Ready for Production
**Last Updated:** 2025-09-29
**Version:** 1.0.0
