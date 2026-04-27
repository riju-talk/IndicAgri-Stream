# Global configuration for IndicAgri-Stream

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPICS = {
    "satellite": "agri.satellite.raw",
    "weather": "agri.weather.raw",
    "iot": "agri.iot.raw",
    "reports": "agri.reports.raw",
    "market": "agri.market.raw",
    "alerts": "agri.alerts.output",
}

# Database Configuration
DATABASE_URL = "postgresql://airflow:airflow@localhost:5432/airflow"
REDIS_URL = "redis://localhost:6379/0"

# AWS Configuration
AWS_REGION = "ap-south-1"
S3_BUCKET = "indicagri-data-lake"
SNS_TOPIC_ARN = ""

# Notification Configuration
TWILIO_ACCOUNT_SID = ""
TWILIO_AUTH_TOKEN = ""
TWILIO_WHATSAPP_NUMBER = "+14155238886"

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Monitoring
PROMETHEUS_PORT = 9090
GRAFANA_PORT = 3000
