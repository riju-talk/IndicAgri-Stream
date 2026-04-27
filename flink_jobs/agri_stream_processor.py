"""PyFlink job for real-time agricultural data stream processing."""

from apache_flink.api.common import RestartStrategyStrategies
from apache_flink.api.common.time import Time
from apache_flink.streaming.api.datastream import DataStream
from apache_flink.streaming.api.environment import StreamExecutionEnvironment
from apache_flink.streaming.api.functions import ProcessAllWindowFunction
from apache_flink.streaming.api.windowing import TumblingEventTimeWindows


class AgriAlertFunction(ProcessAllWindowFunction):
    """Process windowed agricultural data and generate alerts."""

    def process(self, context, elements, out):
        """Process elements in the window and emit alerts."""
        # Calculate average NDVI, moisture, temperature
        ndvi_values = []
        moisture_values = []
        temp_values = []
        
        for event in elements:
            data = event.get("data", {})
            if "ndvi" in data:
                ndvi_values.append(data["ndvi"])
            if "moisture_pct" in data:
                moisture_values.append(data["moisture_pct"])
            if "temperature_c" in data:
                temp_values.append(data["temperature_c"])
        
        # Generate alert conditions
        alerts = []
        plot_id = context.window.max_timestamp()  # Simplified
        
        if ndvi_values:
            avg_ndvi = sum(ndvi_values) / len(ndvi_values)
            if avg_ndvi < 0.3:
                alerts.append({
                    "alert_type": "low_vegetation_health",
                    "severity": "high",
                    "message": f"Low NDVI detected: {avg_ndvi:.2f}",
                })
        
        if moisture_values:
            avg_moisture = sum(moisture_values) / len(moisture_values)
            if avg_moisture < 25:
                alerts.append({
                    "alert_type": "low_soil_moisture",
                    "severity": "critical",
                    "message": f"Irrigation needed: moisture at {avg_moisture:.1f}%",
                })
        
        if temp_values:
            avg_temp = sum(temp_values) / len(temp_values)
            if avg_temp > 38:
                alerts.append({
                    "alert_type": "heat_stress",
                    "severity": "medium",
                    "message": f"Heat stress warning: {avg_temp:.1f}°C",
                })
        
        for alert in alerts:
            out.collect({
                "plot_id": str(plot_id),
                "window_end": context.window.max_timestamp(),
                **alert,
            })


def create_processing_pipeline():
    """Create the Flink streaming pipeline."""
    env = StreamExecutionEnvironment.get_execution_environment()
    
    # Set restart strategy for fault tolerance
    env.set_restart_strategy(
        RestartStrategyStrategies.fixed_delay_restart(
            num_attempts=3,
            delay_between_attempts=Time.seconds(10),
        )
    )
    
    # Enable checkpointing for state management
    env.enable_checkpointing(60000)  # Checkpoint every 60 seconds
    
    # Source: Kafka topics (placeholder - would use FlinkKafkaConsumer)
    # In production: FlinkKafkaConsumer with Protobuf deserialization
    weather_stream: DataStream = env.from_collection([])  # Placeholder
    satellite_stream: DataStream = env.from_collection([])  # Placeholder
    iot_stream: DataStream = env.from_collection([])  # Placeholder
    
    # Union all streams
    all_streams = weather_stream.union(satellite_stream, iot_stream)
    
    # Assign timestamps and watermarks for event-time processing
    # all_streams = all_streams.assign_timestamps_and_watermarks(...)
    
    # Key by plot_id for per-farm processing
    keyed_stream = all_streams.key_by(lambda x: x.get("plot_id", "unknown"))
    
    # Apply tumbling windows (1-hour windows for near-real-time alerts)
    windowed_stream = keyed_stream.window(TumblingEventTimeWindows.of(Time.hours(1)))
    
    # Process windows and generate alerts
    alerts = windowed_stream.apply(AgriAlertFunction())
    
    # Sink: Write alerts to Kafka topic or database
    # In production: FlinkKafkaProducer or JdbcSink
    alerts.print()  # For debugging
    
    return env


def main():
    """Main entry point for Flink job."""
    env = create_processing_pipeline()
    env.execute("Agricultural Stream Processing Job")


if __name__ == "__main__":
    main()
