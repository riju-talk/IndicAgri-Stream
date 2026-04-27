"""FastAPI + GraphQL serving layer for agricultural intelligence."""

from typing import Any, Optional

import strawberry
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Pydantic models for REST API
class PlotStatus(BaseModel):
    """Plot status response model."""

    plot_id: str
    crop_type: Optional[str] = None
    health_score: float = Field(ge=0.0, le=1.0)
    last_updated: str
    alerts: list[dict[str, Any]] = []
    recommendations: list[str] = []


class AlertRequest(BaseModel):
    """Alert creation request model."""

    plot_id: str
    alert_type: str
    severity: str
    message: str


# Strawberry GraphQL types
@strawberry.type
class PlotInfo:
    """GraphQL type for plot information."""

    plot_id: strawberry.ID
    state: str
    district: str
    crop_type: str
    area_hectares: float
    farmer_name: Optional[str] = None


@strawberry.type
class WeatherData:
    """GraphQL type for weather information."""

    temperature_c: float
    humidity_pct: float
    precipitation_mm: float
    forecast_3d: str


@strawberry.type
class SoilHealth:
    """GraphQL type for soil health metrics."""

    moisture_pct: float
    ph: float
    nitrogen_ppm: float
    phosphorus_ppm: float
    potassium_ppm: float
    health_rating: str


@strawberry.type
class AgriculturalAlert:
    """GraphQL type for farm alerts."""

    alert_id: strawberry.ID
    plot_id: str
    alert_type: str
    severity: str
    message: str
    timestamp: str
    acknowledged: bool = False


@strawberry.type
class Query:
    """GraphQL query root."""

    @strawberry.field
    def plot(self, plot_id: strawberry.ID) -> Optional[PlotInfo]:
        """Get plot information by ID."""
        # Placeholder - would query database
        if plot_id:
            return PlotInfo(
                plot_id=plot_id,
                state="WB",
                district="Murshidabad",
                crop_type="rice",
                area_hectares=1.5,
                farmer_name="Ramesh Kumar",
            )
        return None

    @strawberry.field
    def plot_status(self, plot_id: strawberry.ID) -> Optional[PlotStatus]:
        """Get real-time plot status."""
        # Placeholder - would aggregate from multiple sources
        if plot_id:
            return PlotStatus(
                plot_id=plot_id,
                crop_type="rice",
                health_score=0.72,
                last_updated="2024-01-15T10:30:00Z",
                alerts=[
                    {"type": "low_soil_moisture", "severity": "high"},
                ],
                recommendations=[
                    "Irrigate within 2 days",
                    "Monitor for pest activity",
                ],
            )
        return None

    @strawberry.field
    def weather(self, plot_id: str) -> Optional[WeatherData]:
        """Get current weather for a plot."""
        # Placeholder - would fetch from weather service
        return WeatherData(
            temperature_c=32.5,
            humidity_pct=68.0,
            precipitation_mm=2.3,
            forecast_3d="Partly cloudy with 30% chance of rain",
        )

    @strawberry.field
    def soil_health(self, plot_id: str) -> Optional[SoilHealth]:
        """Get soil health metrics for a plot."""
        # Placeholder - would query sensor data
        return SoilHealth(
            moisture_pct=42.5,
            ph=6.8,
            nitrogen_ppm=285.0,
            phosphorus_ppm=28.5,
            potassium_ppm=220.0,
            health_rating="good",
        )

    @strawberry.field
    def alerts(self, plot_id: Optional[str] = None, limit: int = 10) -> list[AgriculturalAlert]:
        """Get recent alerts, optionally filtered by plot."""
        # Placeholder - would query alerts database
        return [
            AgriculturalAlert(
                alert_id="alert_001",
                plot_id=plot_id or "WB_123_45",
                alert_type="pest_sighting",
                severity="high",
                message="Bollworm detected in cotton crop",
                timestamp="2024-01-15T09:15:00Z",
                acknowledged=False,
            ),
        ]


@strawberry.type
class Mutation:
    """GraphQL mutation root."""

    @strawberry.mutation
    def acknowledge_alert(self, alert_id: strawberry.ID) -> bool:
        """Acknowledge an alert."""
        # Placeholder - would update database
        return True

    @strawberry.mutation
    def create_manual_report(
        self,
        plot_id: str,
        report_type: str,
        description: str,
        severity: str = "medium",
    ) -> AgriculturalAlert:
        """Create a manual farmer report."""
        # Placeholder - would save to database and trigger processing
        return AgriculturalAlert(
            alert_id=f"manual_{plot_id}_{len(description)}",
            plot_id=plot_id,
            alert_type=report_type,
            severity=severity,
            message=description,
            timestamp="2024-01-15T10:00:00Z",
            acknowledged=False,
        )


# Create GraphQL schema
schema = strawberry.Schema(query=Query, mutation=Mutation)

# Create FastAPI app
app = FastAPI(
    title="IndicAgri-Stream API",
    description="Real-time agricultural intelligence API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# GraphQL endpoint
@app.post("/graphql")
async def graphql_endpoint(request: strawberry.http.Request):
    """Handle GraphQL requests."""
    result = await schema.execute(
        request.query,
        variable_values=request.variables,
        context_value=request,
    )

    if result.errors:
        raise HTTPException(status_code=400, detail=str(result.errors))

    return result.data


# REST API endpoints
@app.get("/plot/{plot_id}/status", response_model=PlotStatus)
async def get_plot_status(plot_id: str):
    """Get plot status via REST API."""
    status = Query().plot_status(plot_id)
    if not status:
        raise HTTPException(status_code=404, detail="Plot not found")
    return status


@app.get("/plot/{plot_id}/weather")
async def get_plot_weather(plot_id: str):
    """Get weather data for a plot."""
    weather = Query().weather(plot_id)
    return weather


@app.get("/plot/{plot_id}/soil")
async def get_plot_soil(plot_id: str):
    """Get soil health for a plot."""
    soil = Query().soil_health(plot_id)
    return soil


@app.get("/alerts")
async def get_alerts(plot_id: Optional[str] = None, limit: int = 10):
    """Get recent alerts."""
    return Query().alerts(plot_id=plot_id, limit=limit)


@app.post("/alerts", response_model=AgriculturalAlert)
async def create_alert(alert: AlertRequest):
    """Create a new alert."""
    return AgriculturalAlert(
        alert_id=f"alert_{alert.plot_id}_{len(alert.message)}",
        plot_id=alert.plot_id,
        alert_type=alert.alert_type,
        severity=alert.severity,
        message=alert.message,
        timestamp="2024-01-15T10:00:00Z",
        acknowledged=False,
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "indicagri-stream-api"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
