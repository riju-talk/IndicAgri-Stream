"""Mock event generator for agricultural data pipeline.

Generates synthetic events for satellite imagery, weather, soil sensors,
farmer reports, and market prices to simulate real-time data ingestion.
"""

import argparse
import json
import random
import uuid
from datetime import datetime, timedelta
from typing import Any

# Mock data constants
INDIAN_STATES = ["WB", "UP", "MH", "KA", "PB", "RJ", "MP", "GJ", "TN", "AP"]
CROPS = ["rice", "wheat", "sugarcane", "cotton", "maize", "mustard", "groundnut"]
PEST_TYPES = ["locust", "bollworm", "stem_borer", "aphid", "blast"]
SOIL_TYPES = ["clay", "loam", "sandy", "silt"]


def generate_plot_id(state: str | None = None) -> str:
    """Generate a unique plot ID."""
    if state is None:
        state = random.choice(INDIAN_STATES)
    return f"{state}_{random.randint(100, 999)}_{random.randint(1, 500)}"


def generate_satellite_event(plot_id: str, timestamp: datetime) -> dict[str, Any]:
    """Generate Sentinel-2 satellite observation event."""
    return {
        "event_uuid": str(uuid.uuid4()),
        "event_type": "satellite_observation",
        "timestamp": timestamp.isoformat(),
        "plot_id": plot_id,
        "data": {
            "ndvi": round(random.uniform(0.2, 0.8), 3),
            "evi": round(random.uniform(0.15, 0.6), 3),
            "cloud_cover": round(random.uniform(0, 0.4), 2),
            "resolution_m": 10,
            "band_combination": "B8A_B4_B3",
        },
        "metadata": {
            "satellite": "Sentinel-2",
            "processing_level": "L2A",
        },
    }


def generate_weather_event(plot_id: str, timestamp: datetime) -> dict[str, Any]:
    """Generate IMD/OpenWeather API weather event."""
    return {
        "event_uuid": str(uuid.uuid4()),
        "event_type": "weather_observation",
        "timestamp": timestamp.isoformat(),
        "plot_id": plot_id,
        "data": {
            "temperature_c": round(random.uniform(20, 40), 1),
            "humidity_pct": round(random.uniform(40, 90), 1),
            "precipitation_mm": round(random.uniform(0, 50), 2),
            "wind_speed_kmh": round(random.uniform(5, 30), 1),
            "forecast_3d": {
                "temp_max": round(random.uniform(25, 42), 1),
                "rain_probability": round(random.uniform(0, 1), 2),
            },
        },
        "metadata": {
            "source": "IMD",
            "station_id": f"STN_{random.randint(1000, 9999)}",
        },
    }


def generate_soil_sensor_event(plot_id: str, timestamp: datetime) -> dict[str, Any]:
    """Generate IoT soil sensor reading event."""
    return {
        "event_uuid": str(uuid.uuid4()),
        "event_type": "soil_sensor_reading",
        "timestamp": timestamp.isoformat(),
        "plot_id": plot_id,
        "data": {
            "moisture_pct": round(random.uniform(20, 60), 1),
            "ph": round(random.uniform(5.5, 7.5), 2),
            "nitrogen_ppm": round(random.uniform(200, 400), 1),
            "phosphorus_ppm": round(random.uniform(15, 40), 1),
            "potassium_ppm": round(random.uniform(150, 300), 1),
            "temperature_c": round(random.uniform(22, 32), 1),
        },
        "metadata": {
            "sensor_id": f"SENSOR_{random.randint(10000, 99999)}",
            "battery_level": round(random.uniform(60, 100), 1),
            "soil_type": random.choice(SOIL_TYPES),
        },
    }


def generate_farmer_report_event(plot_id: str, timestamp: datetime) -> dict[str, Any]:
    """Generate farmer voice/text report event."""
    languages = ["hi", "bn", "mr", "ta", "te", "gu", "pa", "en"]
    report_types = ["pest_sighting", "disease_symptom", "irrigation_issue", "harvest_ready"]
    
    return {
        "event_uuid": str(uuid.uuid4()),
        "event_type": "farmer_report",
        "timestamp": timestamp.isoformat(),
        "plot_id": plot_id,
        "data": {
            "report_type": random.choice(report_types),
            "description": f"Observed {random.choice(PEST_TYPES)} on {random.choice(CROPS)}",
            "language": random.choice(languages),
            "severity": random.choice(["low", "medium", "high", "critical"]),
            "image_url": f"s3://agri-images/{plot_id}/{uuid.uuid4()}.jpg" if random.random() > 0.5 else None,
        },
        "metadata": {
            "farmer_id": f"FARMER_{random.randint(1000, 9999)}",
            "channel": random.choice(["whatsapp", "sms", "voice_call"]),
            "location": {"lat": round(random.uniform(20, 30), 4), "lon": round(random.uniform(80, 90), 4)},
        },
    }


def generate_market_price_event(crop: str | None = None, timestamp: datetime | None = None) -> dict[str, Any]:
    """Generate e-NAM market price event."""
    if crop is None:
        crop = random.choice(CROPS)
    if timestamp is None:
        timestamp = datetime.now()
    
    markets = ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bangalore", "Hyderabad", "Ahmedabad", "Pune"]
    
    return {
        "event_uuid": str(uuid.uuid4()),
        "event_type": "market_price_update",
        "timestamp": timestamp.isoformat(),
        "crop": crop,
        "data": {
            "price_per_quintal_inr": round(random.uniform(1500, 4500), 2),
            "arrival_quantity_mt": round(random.uniform(50, 500), 1),
            "price_change_pct": round(random.uniform(-5, 5), 2),
            "market_location": random.choice(markets),
        },
        "metadata": {
            "source": "e-NAM",
            "market_yard_id": f"YARD_{random.randint(100, 999)}",
        },
    }


def generate_events(num_plots: int = 50, num_days: int = 7, output_format: str = "json") -> list[dict[str, Any]]:
    """Generate a sequence of mock agricultural events."""
    events = []
    base_time = datetime.now() - timedelta(days=num_days)
    
    # Generate plot IDs
    plot_ids = [generate_plot_id() for _ in range(num_plots)]
    
    # Generate events for each day
    for day_offset in range(num_days):
        current_day = base_time + timedelta(days=day_offset)
        
        # Generate multiple events per day per plot
        for plot_id in plot_ids:
            # Satellite pass (once every 2-3 days per plot)
            if day_offset % 3 == 0 or random.random() > 0.6:
                events.append(generate_satellite_event(plot_id, current_day + timedelta(hours=random.randint(10, 14))))
            
            # Weather updates (every 6 hours)
            for hour in [0, 6, 12, 18]:
                events.append(generate_weather_event(plot_id, current_day + timedelta(hours=hour)))
            
            # Soil sensor readings (every 2 hours during daylight)
            for hour in range(6, 19, 2):
                events.append(generate_soil_sensor_event(plot_id, current_day + timedelta(hours=hour)))
            
            # Farmer reports (random, ~10% chance per day)
            if random.random() < 0.1:
                events.append(generate_farmer_report_event(plot_id, current_day + timedelta(hours=random.randint(8, 18))))
        
        # Market price updates (daily for each crop)
        for crop in CROPS:
            events.append(generate_market_price_event(crop, current_day + timedelta(hours=12)))
    
    # Sort by timestamp
    events.sort(key=lambda x: x["timestamp"])
    
    return events


def main():
    """Main entry point for mock event generation."""
    parser = argparse.ArgumentParser(description="Generate mock agricultural events")
    parser.add_argument("--plots", type=int, default=50, help="Number of farm plots")
    parser.add_argument("--days", type=int, default=7, help="Number of days of data")
    parser.add_argument("--output", choices=["json", "kafka"], default="json", help="Output format")
    parser.add_argument("--output-file", type=str, help="Output file path (default: stdout)")
    args = parser.parse_args()
    
    print(f"Generating mock events for {args.plots} plots over {args.days} days...", flush=True)
    events = generate_events(num_plots=args.plots, num_days=args.days, output_format=args.output)
    
    if args.output == "json":
        output = json.dumps(events, indent=2)
        if args.output_file:
            with open(args.output_file, "w") as f:
                f.write(output)
            print(f"Events written to {args.output_file}", flush=True)
        else:
            print(output)
    
    elif args.output == "kafka":
        # For Kafka output, we'd integrate with confluent-kafka producer
        # This is a placeholder for the actual implementation
        print("Kafka output mode selected - would publish to Kafka topics:", flush=True)
        topic_map = {
            "satellite_observation": "agri.satellite.raw",
            "weather_observation": "agri.weather.raw",
            "soil_sensor_reading": "agri.iot.raw",
            "farmer_report": "agri.reports.raw",
            "market_price_update": "agri.market.raw",
        }
        
        for event in events:
            topic = topic_map.get(event["event_type"], "agri.misc.raw")
            print(f"[{topic}] {event['event_uuid'][:8]}... - {event['event_type']} for {event.get('plot_id', 'N/A')}", flush=True)
    
    print(f"\nGenerated {len(events)} total events", flush=True)


if __name__ == "__main__":
    main()
