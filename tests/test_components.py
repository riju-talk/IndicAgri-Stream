"""Unit tests for agricultural stream processing components."""

import pytest
from datetime import datetime, timedelta


class TestMockEventGenerator:
    """Tests for mock event generation."""

    def test_generate_plot_id(self):
        """Test plot ID generation format."""
        from scripts.generate_mock_events import generate_plot_id
        
        plot_id = generate_plot_id()
        parts = plot_id.split("_")
        
        assert len(parts) == 3
        assert len(parts[0]) == 2  # State code
        assert parts[1].isdigit()
        assert parts[2].isdigit()

    def test_generate_satellite_event(self):
        """Test satellite event structure."""
        from scripts.generate_mock_events import generate_satellite_event
        
        timestamp = datetime.now()
        event = generate_satellite_event("WB_123_45", timestamp)
        
        assert event["event_type"] == "satellite_observation"
        assert event["plot_id"] == "WB_123_45"
        assert "event_uuid" in event
        assert "data" in event
        assert "ndvi" in event["data"]
        assert 0.2 <= event["data"]["ndvi"] <= 0.8

    def test_generate_weather_event(self):
        """Test weather event structure."""
        from scripts.generate_mock_events import generate_weather_event
        
        timestamp = datetime.now()
        event = generate_weather_event("UP_456_78", timestamp)
        
        assert event["event_type"] == "weather_observation"
        assert "temperature_c" in event["data"]
        assert "humidity_pct" in event["data"]
        assert 20 <= event["data"]["temperature_c"] <= 40

    def test_generate_events_count(self):
        """Test total event count generation."""
        from scripts.generate_mock_events import generate_events
        
        events = generate_events(num_plots=5, num_days=2)
        assert len(events) > 0
        
        # Verify sorting by timestamp
        timestamps = [e["timestamp"] for e in events]
        assert timestamps == sorted(timestamps)


class TestNotificationService:
    """Tests for notification service."""

    def test_notification_config_defaults(self):
        """Test default notification configuration."""
        from src.notifications.alert_service import NotificationConfig
        
        config = NotificationConfig()
        assert config.default_language == "en"
        assert config.aws_sns_region == "ap-south-1"

    def test_alert_message_creation(self):
        """Test alert message model."""
        from src.notifications.alert_service import AlertMessage
        
        alert = AlertMessage(
            alert_id="test_001",
            plot_id="WB_123_45",
            farmer_id="FARMER_001",
            alert_type="pest_sighting",
            severity="high",
            message_en="Pest detected in crop",
        )
        
        assert alert.channel == "whatsapp"
        assert alert.language == "en"

    def test_translator_english_passthrough(self):
        """Test translator returns English unchanged."""
        from src.notifications.alert_service import IndicNLPTranslator
        
        translator = IndicNLPTranslator()
        message = "Irrigation needed immediately"
        
        result = translator.translate(message, "en")
        assert result == message

    def test_format_alert_message(self):
        """Test alert message formatting."""
        from src.notifications.alert_service import NotificationService, AlertMessage
        
        service = NotificationService()
        alert = AlertMessage(
            alert_id="test_001",
            plot_id="WB_123_45",
            farmer_id="FARMER_001",
            alert_type="low_soil_moisture",
            severity="critical",
            message_en="Soil moisture critically low",
        )
        
        formatted = service.format_alert_message(alert)
        assert "🆘" in formatted  # Critical severity emoji
        assert "WB_123_45" in formatted
        assert "low soil moisture" in formatted.lower()


class TestAPIEndpoints:
    """Tests for FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from src.serving.api import app
        return TestClient(app)

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_get_plot_status(self, client):
        """Test plot status endpoint."""
        response = client.get("/plot/WB_123_45/status")
        assert response.status_code == 200
        data = response.json()
        assert data["plot_id"] == "WB_123_45"
        assert "health_score" in data

    def test_get_alerts(self, client):
        """Test alerts endpoint."""
        response = client.get("/alerts?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_create_alert(self, client):
        """Test alert creation endpoint."""
        alert_data = {
            "plot_id": "UP_789_12",
            "alert_type": "weather_alert",
            "severity": "medium",
            "message": "Heavy rain forecasted",
        }
        response = client.post("/alerts", json=alert_data)
        assert response.status_code == 200
        data = response.json()
        assert data["alert_type"] == "weather_alert"


class TestGraphQLSchema:
    """Tests for GraphQL schema."""

    @pytest.mark.asyncio
    async def test_plot_query(self):
        """Test GraphQL plot query."""
        from src.serving.api import schema
        
        query = """
            query {
                plot(plotId: "WB_123_45") {
                    plotId
                    state
                    district
                    cropType
                }
            }
        """
        
        result = await schema.execute(query)
        assert result.errors is None
        assert result.data["plot"]["plotId"] == "WB_123_45"

    @pytest.mark.asyncio
    async def test_weather_query(self):
        """Test GraphQL weather query."""
        from src.serving.api import schema
        
        query = """
            query {
                weather(plotId: "WB_123_45") {
                    temperatureC
                    humidityPct
                    precipitationMm
                }
            }
        """
        
        result = await schema.execute(query)
        assert result.errors is None
        assert "temperatureC" in result.data["weather"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
